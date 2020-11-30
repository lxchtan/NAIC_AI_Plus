import torch
from torch.utils.data import TensorDataset, DataLoader
import models
from pathlib import Path
from ignite.utils import setup_logger
from ignite.handlers import Checkpoint, DiskSaver
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

from dataset import set_seed

def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--model", type=str, default='ffn', help="model's name")
  parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file path")
  parser.add_argument("--pilot_version", type=int, choices=[1, 2], default=1)
  parser.add_argument("--batch_size", type=int, default=1024)
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training (-1: not distributed)")
  parser.add_argument("--seed", type=int, default=43)
  parser.add_argument("--debug", action='store_true') 
  args = parser.parse_args()

  # Setup CUDA, GPU & distributed training
  args.distributed = (args.local_rank != -1)
  if not args.distributed:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
  args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
  args.device = device

  # Set seed
  set_seed(args)
  logger = setup_logger("Testing", distributed_rank=args.local_rank)

  # Model construction
  model = getattr(models, args.model)(args)
  checkpoint_fp = Path(args.checkpoint)
  assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
  logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
  checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
  Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
  model = model.to(device)

  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  datapath = f'data/Y_{args.pilot_version}.csv'
  dataY = pd.read_csv(datapath, header=None).values

  test_dataset = torch.tensor(dataY, dtype=torch.float32)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

  pred = []
  model.eval()
  for batch in tqdm(test_loader, desc="Runing Testing"):
    batch = batch.to(device)
    x_pred = model(batch)
    x_pred = x_pred > 0.5
    pred.append(x_pred.cpu().numpy())
  np.concatenate(pred).tofile(f'{os.path.split(args.checkpoint)[0]}/X_pre_{args.pilot_version}.bin')
  if args.debug:
    np.ones_like(np.concatenate(pred)).tofile(f'{os.path.split(args.checkpoint)[0]}/X_pre_2.bin') 

if __name__ == "__main__":
  main()