from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import numpy as np
from numpy import random
import random
import os
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from utils import MIMO
import struct
from itertools import islice
from threading import Thread
import gc
from ignite.utils import convert_tensor

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# ########产生测评数据，仅供参考格式##########

class SIGDataset(torch.utils.data.Dataset):
  def __init__(self, args, data_type="train"):
    # Pilotnum=32, reset=False, mode=None, SNR=None, datapath='data/H.bin'
    mode = args.mode
    SNR = args.SNRdb
    reset = False
    pilot_num = 32 if args.pilot_version == 1 else 8
    datapath = os.path.join(args.data_dir, "H.bin")
    train_cache = args.cache_dir

    mode_str = f"_mode_{mode}" if mode is not None else ""
    SNR_str  = f"_SNR_{SNR:.1f}" if SNR is not None else ""
    # nn_str = "_nn" if args.with_pure_y else ""
    nn_str = "_nn"
    if data_type == "train":
      self.cache_num = len(list(
        filter(
          lambda x: x.startswith(f'{data_type}_P{pilot_num}{mode_str}{SNR_str}_epoch') and x.endswith(f'dataset{nn_str}.pkl'), 
          os.listdir(train_cache)
        )
      ))
      print(f"We have {self.cache_num} cache dataset.")
      self.cache_index = args.first_cache_index % self.cache_num
      self.train_cache = train_cache

    self.Pilotnum = pilot_num #8
    self.datapath = datapath
    H, Htest = self.read_data()
    self.data_type = data_type
    self.H = H if data_type == "train" else Htest
    self.data_len = len(self.H)
    self.mode = mode
    self.SNRdb = SNR
    self.mode_str = mode_str
    self.SNR_str = SNR_str
    self.nn_str = nn_str
    self.with_pure_y = args.with_pure_y
    
    if not args.no_cache and data_type == "train":
      self.reading_cache_thread = Thread(target=self.reading_cache)
      self.reading_cache_thread.start()

    dataset_cache = os.path.split(datapath)[0] + f'/{data_type}_P{pilot_num}{mode_str}{SNR_str}_dataset{nn_str}.pkl'
    if not os.path.exists(dataset_cache):
      reset = True
    if reset:
      self.reset()
      self.save_one_cache(dataset_cache)
    else:
      self.read_one_cache(dataset_cache)
  
  def save_one_cache(self, dataset_cache):
    _data = ([x.to(torch.bool) for x in self.tensorX], self.tensorY, self.modes, self.SNRdbs, self.tensorH)
    if self.with_pure_y:
      _data += (self.pureY,)
    torch.save(_data, dataset_cache)

  def read_one_cache(self, dataset_cache, temp=False):
    _data = torch.load(dataset_cache)
    if not temp:
      self.tensorX, self.tensorY, self.modes, self.SNRdbs, self.tensorH = _data[:5]
      if self.with_pure_y:
        self.pureY = _data[5]
      self.tensorX = [x.to(torch.float32) for x in self.tensorX]
    else:
      self.tensorX_, self.tensorY_, self.modes_, self.SNRdbs_, self.tensorH_ = _data[:5]
      if self.with_pure_y:
        self.pureY_ = _data[5]
      self.tensorX_ = [x.to(torch.float32) for x in self.tensorX_]

  def exchange_data(self):
    # del self.tensorX, self.tensorY, self.modes, self.SNRdbs
    # gc.collect()
    self.tensorX, self.tensorY, self.modes, self.SNRdbs, self.tensorH = self.tensorX_, self.tensorY_, self.modes_, self.SNRdbs_, self.tensorH_
    if self.with_pure_y:
      self.pureY = self.pureY_

  def read_data(self):
    data1 = open(self.datapath, 'rb')
    H1 = struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
    H1 = np.reshape(H1, [320000, 2, 4, 32])
    H = H1[:, 1, :, :]+1j*H1[:, 0, :, :]

    Htest = H[300000:, :, :]
    H = H[:300000, :, :]

    return H, Htest

  def generator(self):
    H = self.H
    Pilotnum = self.Pilotnum
    y = []
    y_without_noise = []
    x = []
    modes = []
    SNRdbs = []
    Hs = []
    for _ in trange(self.data_len, desc="Generate Data"):
      mode = random.randint(0, 3) if self.mode is None else self.mode
      SNRdb = random.uniform(8, 12)  if self.SNRdb is None else self.SNRdb
      bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
      bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
      X = [bits0, bits1]
      temp = np.random.randint(0, len(H))
      HH = H[temp]
      batch_y = MIMO(X, HH, SNRdb, mode, Pilotnum)/20
      batch_y_without_noise = MIMO(X, HH, 2e3, mode, Pilotnum)/20
      batch_x = np.concatenate((bits0, bits1), 0)
      y.append(batch_y.astype(np.float32))
      y_without_noise.append(batch_y_without_noise.astype(np.float32))
      x.append(batch_x.astype(np.float32))
      modes.append(mode)
      SNRdbs.append(SNRdb)
      myH = np.concatenate((np.expand_dims(np.real(HH), 1), np.expand_dims(np.imag(HH), 1)), axis=1)
      Hs.append(myH.astype(np.float32))
    return y, x, modes, SNRdbs, Hs, y_without_noise

  def reset(self):
    ty, tx, modes, SNRdbs, Hs, y_without_noise = self.generator()
    self.tensorH = [torch.tensor(H) for H in Hs]
    self.tensorY = [torch.tensor(y) for y in ty]
    self.pureY = [torch.tensor(yn) for yn in y_without_noise]
    self.tensorX = [torch.tensor(x) for x in tx] 
    self.modes = torch.tensor(modes)
    self.SNRdbs = torch.tensor(SNRdbs)

  def reading_cache(self):
    self.cache_index = (self.cache_index + 1) % self.cache_num
    dataset_cache = f'{self.data_type}_P{self.Pilotnum}{self.mode_str}{self.SNR_str}_epoch{self.cache_index:03d}_dataset{self.nn_str}.pkl'
    dataset_cache = os.path.join(self.train_cache, dataset_cache)
    self.read_one_cache(dataset_cache, temp=True)

  def reload(self):
    self.reading_cache_thread.join()
    print(f'{self.data_type}_P{self.Pilotnum}{self.mode_str}{self.SNR_str}_epoch{self.cache_index:03d}_dataset{self.nn_str}.pkl')
    self.exchange_data()
    self.reading_cache_thread = Thread(target=self.reading_cache)
    self.reading_cache_thread.start()

  @staticmethod
  def prepare_batch(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
  ):
    """Prepare batch for training: pass to a device with options.

    """
    x, y, x_pure, H = batch

    # y = convert_tensor(y, device=device, non_blocking=non_blocking)
    # y = torch.fft.ifft(0.7071 * (2 * y[:, ::2] - 1)+ 0.7071j * (2 * y[:, 1::2] - 1), dim=-1)
    # data_y = torch.stack([y.real, y.imag], dim=-1).to(dtype=torch.float32).reshape(y.size(0), -1)
    # data_y = 0.7071 * (2 * y - 1)

    H_tensor = convert_tensor(H, device=device, non_blocking=non_blocking)
    # dataH_ri = torch.fft.fft(
    #   torch.cat([H_tensor[:, :, 0, :] + 1j * H_tensor[:, :, 1, :], torch.zeros(H_tensor.size(0), 4, 256 - 32, dtype=torch.complex64, device=device)], dim=-1)
    # )
    # dataH = torch.stack([dataH_ri.real.to(dtype=torch.float32), dataH_ri.imag.to(dtype=torch.float32)], dim=2) / 20
    dataH = H_tensor / 20
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
        convert_tensor(x_pure, device=device, non_blocking=non_blocking),
        dataH,
    )

  def __getitem__(self, index):
    dataX = self.tensorX[index]
    dataY = self.tensorY[index]
    H = self.tensorH[index]
    dataH = H

    if self.with_pure_y:
      dataYN = self.pureY[index]
    else:
      dataYN = torch.zeros_like(dataY)

    return dataY, dataX, dataYN, dataH

  def __len__(self):
    return self.data_len

# class SIGDataset(torch.utils.data.IterableDataset):
#   def __init__(self, datapath='data/H.bin', Pilotnum=32, data_type="train"):
#     self.Pilotnum = Pilotnum #8
#     self.datapath = datapath
#     H, Htest = self.read_data()
#     self.data_type = data_type
#     self.H = H if data_type == "train" else Htest

#   def read_data(self):
#     data1 = open(self.datapath, 'rb')
#     H1 = struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
#     H1 = np.reshape(H1, [320000, 2, 4, 32])
#     H = H1[:, 1, :, :]+1j*H1[:, 0, :, :]

#     Htest = H[300000:, :, :]
#     H = H[:300000, :, :]

#     return H, Htest

#   def generator(self):
#     H = self.H
#     Pilotnum = self.Pilotnum
#     while True:
#       mode = random.randint(0, 3)
#       SNRdb = random.uniform(8, 12)
#       bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
#       bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
#       X = [bits0, bits1]
#       temp = np.random.randint(0, len(H))
#       HH = H[temp]
#       batch_y = MIMO(X, HH, SNRdb, mode, Pilotnum)/20
#       batch_x = np.concatenate((bits0, bits1), 0)
#       yield (batch_y.astype(np.float32), batch_x.astype(np.float32))

#   def __iter__(self):
#     return self.generator()

#   def __len__(self):
#     return len(self.H) * 2

# def generatorXY(batch, H):
#     input_labels = []
#     input_samples = []
#     for row in range(0, batch):
#         bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
#         bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
#         X = [bits0, bits1]
#         temp = np.random.randint(0, len(H))
#         HH = H[temp]
#         YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20
#         XX = np.concatenate((bits0, bits1), 0)
#         input_labels.append(XX)
#         input_samples.append(YY)
#     batch_y = np.asarray(input_samples)
#     batch_x = np.asarray(input_labels)
#     return batch_y, batch_x


# Y, X = generatorXY(10000, H)
# np.savetxt('Y_1.csv', Y, delimiter=',')
# X_1 = np.array(np.floor(X + 0.5), dtype=np.bool)
# X_1.tofile('X_1.bin')
