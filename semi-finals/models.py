import torch
import torch.nn.functional as F
from torch import nn
import torch.fft
from utils import get_pilot_carriers

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
      nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True),
      nn.BatchNorm2d(out_planes)
    )

def conv3x3_3D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
      nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True),
      nn.BatchNorm3d(out_planes)
    )

def conv1d_ln(in_planes, out_planes, kernel_size=3, stride=1, padding=0):
    """3x3 convolution with padding"""
    return nn.Sequential(
      nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=True),
      nn.BatchNorm1d(out_planes)
    )

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
      nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=True),
      nn.BatchNorm2d(out_planes)
    )

class ffn_concat(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.projections = nn.ModuleList([
      nn.Sequential(
        nn.Linear(2048, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 16),
      ) for _ in range(64)
    ])
    self.sig = nn.Sigmoid()
  
  def forward(self, input, H=None):
    output = torch.cat([proj(input) for proj in self.projections], dim=-1)
    output = self.sig(output)

    return output

class denoise_(nn.Module):
  def __init__(self, image_channels) -> None:
    super().__init__()
    layers = []
    depth=17
    n_channels=64
    padding=3
    kernel_size=7
    layers.append(nn.Conv1d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
    layers.append(nn.ReLU(inplace=True))
    for _ in range(depth-2):
        layers.append(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm1d(n_channels, eps=0.0001, momentum = 0.95))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv1d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
    self.denoise = nn.Sequential(*layers)

  
  def forward(self, input):
    output = self.denoise(input)
    return output

class YHX_linear(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    pilot_num = 32 if args.pilot_version == 1 else 8
    _pilot, _pindex, _pvalue = get_pilot_carriers(pilot_num)
    pindex = torch.tensor(_pindex, device=args.device, requires_grad=False).view(2, 1, -1).repeat(1, 2, 1).reshape(4, -1)
    self.pvalue = torch.tensor(_pvalue, device=args.device, requires_grad=False).unsqueeze(0)
    self.index_ = (pindex + torch.tensor([0, 256, 0, 256], dtype=pindex.dtype, device=args.device).unsqueeze(1)).reshape(1, -1)

    self.one_h = self.pvalue.size(-1) // 4
    self.pieces = 256 // self.one_h

    self.y_denoise = nn.ModuleList([denoise_(image_channels=2) for _ in range(2)])
    self.h_noise_fix = nn.ModuleList([denoise_(image_channels=2) for _ in range(4)])

    self.projections1 = nn.ModuleList([
      nn.Sequential(
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 16),
      ) for _ in range(64)
    ])
    self.sig = nn.Sigmoid()
  
  def forward(self, input, pure=None, H=None, opp=False):
    bz = input.size(0)
    input_ = input.view(bz, -1, 2, 2, 2) 
    index_ = self.index_
    pvalue = self.pvalue

    y_pilot, y_data = map(lambda x: x.squeeze(3), input_.split(1, dim=3)) # bz, 256, 2, 2

    # y denoise
    y_pilot = y_pilot.permute(0, 2, 3, 1)
    y_pilot_noise = torch.stack([noise_fix(y_pilot[:, _id, ...])  for _id, noise_fix in enumerate(self.y_denoise)], dim=1)
    y_pilot = y_pilot - y_pilot_noise
    y_pilot = y_pilot.permute(0, 3, 1, 2)

    y_data = y_data.permute(0, 2, 3, 1)
    y_data_noise = torch.stack([noise_fix(y_data[:, _id, ...])  for _id, noise_fix in enumerate(self.y_denoise)], dim=1)
    y_data = y_data - y_data_noise # bz, 2, ri, 256
    y_data = y_data.permute(0, 3, 1, 2)

    y_pilot_c = y_pilot[:, :, :, 0] + 1j * y_pilot[:, :, :, 1]
    y_data_c = y_data[:, :, :, 0] + 1j * y_data[:, :, :, 1]

    y_pilot_c_ = y_pilot_c.permute(0, 2, 1).reshape(bz, -1)
    dummy_ = index_.expand(bz, index_.size(-1))
    H0_cal = y_pilot_c_.detach().gather(dim=-1, index=dummy_) / pvalue 

    # Time fix
    h0_cal = torch.fft.ifft(H0_cal.reshape(bz, 4, -1), dim=-1) # bz, 4, 32
    h0_cal_ri = torch.stack([h0_cal.real.to(dtype=torch.float32), h0_cal.imag.to(dtype=torch.float32)], dim=2)
    h_noise_ri = torch.stack([noise_fix(h0_cal_ri[:, _id, ...])  for _id, noise_fix in enumerate(self.h_noise_fix)], dim=1) 
    h0_noise_fix_ri = h0_cal_ri - h_noise_ri # bz, 4, 2, 32
    h0_noise_fix = h0_noise_fix_ri[:, :, 0, :] + 1j * h0_noise_fix_ri[:, :, 1, :]
    H0_noise_fix = torch.fft.fft(
      torch.cat([h0_noise_fix, 
      torch.zeros(bz, 4, 256 - 32, dtype=torch.complex64, device=h0_noise_fix.device)], 
      dim=-1))
        
    H0_now = H0_noise_fix
    H0_now_inv = H0_now.permute(0, 2, 1).reshape(bz, -1, 2, 2).inverse().to(dtype=torch.complex64)
    x_init = torch.einsum("bcij,bci->bjc", H0_now_inv, y_data_c)
    x_init_ri = torch.stack([x_init.real.to(dtype=torch.float32), x_init.imag.to(dtype=torch.float32)], dim=-1).reshape(bz, -1) # (bz, 2*256*2)
    x_init_ri = x_init_ri.detach()
    
    output = torch.cat([proj(x_init_ri) for proj in self.projections1], dim=-1)
    output = self.sig(output)

    if opp:
      y_pred = torch.stack([y_pilot, y_data], dim=3).reshape(bz, -1)
      output = (output, y_pred, h0_noise_fix_ri)

    return output
