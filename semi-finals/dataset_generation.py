from logging import disable
import torch
import random
import numpy as np
import struct
import argparse
import os
from functools import partial
from utils import MIMO
from tqdm import trange, tqdm
from multiprocessing import Pool, RLock

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def read_data(datapath):
  data1 = open(os.path.join(datapath, "H.bin"), 'rb')
  H1 = struct.unpack('f'*2*2*2*32*320000, data1.read(4*2*2*2*32*320000))
  H1 = np.reshape(H1, [320000, 2, 4, 32])
  H = H1[:, 1, :, :]+1j*H1[:, 0, :, :]

  Htest = H[300000:, :, :]
  H = H[:300000, :, :]

  return H, Htest

def generate_data(rank, args, H):
  set_seed(args.seed + rank)
  pilot_num = 32 if args.pilot_version == 1 else 8
  y = []
  y_without_noise = []
  x = []
  modes = []
  SNRdbs = []
  Hs = []
  mode_str = f"_mode_{args.mode}" if args.mode is not None else ""
  SNR_str  = f"_SNR_{args.SNRdb:.1f}" if args.SNRdb is not None else ""
  for _ in trange(len(H), desc=f"#{rank} Generate Data", position=rank % args.processes):
    mode = random.randint(0, 3) if args.mode is None else args.mode
    SNRdb = random.uniform(8, 12)  if args.SNRdb is None else args.SNRdb
    bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
    bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
    X = [bits0, bits1]
    temp = np.random.randint(0, len(H))
    HH = H[temp]
    batch_y = MIMO(X, HH, SNRdb, mode, pilot_num)/20
    batch_y_without_noise = MIMO(X, HH, 2e3, mode, pilot_num)/20
    batch_x = np.concatenate((bits0, bits1), 0)
    y.append(batch_y.astype(np.float32))
    y_without_noise.append(batch_y_without_noise.astype(np.float32))
    x.append(batch_x.astype(np.float32))
    modes.append(mode)
    SNRdbs.append(SNRdb)
    myH = np.concatenate((np.expand_dims(np.real(HH), 1), np.expand_dims(np.imag(HH), 1)), axis=1)
    Hs.append(myH.astype(np.float32))
  ty, tx, modes, SNRdbs, Hs = y, x, modes, SNRdbs, Hs
  tensorH = [torch.tensor(H) for H in Hs]
  tensorY = [torch.tensor(y) for y in ty]
  tensorYN = [torch.tensor(yn) for yn in y_without_noise]
  tensorX = [torch.tensor(x) for x in tx] 
  modes = torch.tensor(modes)
  SNRdbs = torch.tensor(SNRdbs)

  nn_str = f"_nn" if args.with_pure_y else ""
  dataset_cache = os.path.join(args.datapath, "train_cache", f'{args.data_type}_P{pilot_num}{mode_str}{SNR_str}_epoch{rank:03d}_dataset{nn_str}.pkl')
  _data = ([x.to(torch.bool) for x in tensorX], tensorY, modes, SNRdbs, tensorH)
  if args.with_pure_y:
    _data += (tensorYN,)
  torch.save(_data, dataset_cache)
  tqdm.write(f"Rank {rank} task is finished!")


def main():
  parser = argparse.ArgumentParser()
  # Required parameters
  parser.add_argument("--datapath", type=str, default="data")
  parser.add_argument("--data_type", type=str, default="train")
  parser.add_argument("--pilot_version", type=int, choices=[1, 2], default=1)
  parser.add_argument("--processes", type=int, default=4)
  parser.add_argument("--data_nums", type=int, default=64)
  parser.add_argument("--seed", type=int, default=43)
  parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=None)
  parser.add_argument("--SNRdb", type=float, default=None)
  parser.add_argument("--with_pure_y", action='store_true') 
  parser.add_argument("--debug", action='store_true')
  args = parser.parse_args()

  H, Htest = read_data(args.datapath)
  using_H = H if args.data_type == "train" else Htest
  
  generate_data_fix = partial(generate_data, args=args, H=using_H)

  tqdm.set_lock(RLock())
  with Pool(processes=args.processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
    [pool.map(generate_data_fix, range(args.processes*i, args.processes*(i+1))) for i in range(args.data_nums//args.processes)]

if __name__ == "__main__":
  main()
