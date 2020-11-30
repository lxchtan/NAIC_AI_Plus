from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import nn
from torch.optim import optimizer
from torch.optim import RMSprop, SGD, AdamW
from torch.utils.data import DataLoader
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator 
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LinearCyclicalScheduler, CosineAnnealingScheduler
from ignite.utils import setup_logger, convert_tensor

import argparse

import os
from pathlib import Path
from pprint import pformat

import models
from dataset import SIGDataset, set_seed
import auxiliary

def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--model", type=str, default='ffn', help="model's name")
  parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=None)
  parser.add_argument("--SNRdb", type=float, default=None)
  parser.add_argument("--pilot_version", type=int, choices=[1, 2], default=1)
  parser.add_argument("--loss_type", type=str, default="BCELoss")
  parser.add_argument("--train_batch_size", type=int, default=128)
  parser.add_argument("--valid_batch_size", type=int, default=128)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
  parser.add_argument("--max_norm", type=float, default=-1)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--noise_lambda", type=float, default=1.0)
  parser.add_argument("--lr_scheduler", type=str, choices=["linear", "cycle", "cosine"], default="linear")
  parser.add_argument("--reset_lr_scheduler", type=str, choices=["linear", "cycle", "cosine"], default=None)
  parser.add_argument("--reset_trainer", action='store_true')
  parser.add_argument("--modify_model", action='store_true')
  parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
  parser.add_argument("--eval_iter", type=int, default=10)
  parser.add_argument("--save_iter", type=int, default=10)
  parser.add_argument("--n_epochs", type=int, default=10)
  parser.add_argument("--flush_dataset", type=int, default=0)
  parser.add_argument("--no_cache", action='store_true')
  parser.add_argument("--with_pure_y", action='store_true') 
  parser.add_argument("--with_h", action='store_true') 
  parser.add_argument("--only_l1", action='store_true', help="Only loss 1")
  parser.add_argument("--interpolation", action='store_true', help="if interpolate between pure and reconstruction.") 
  parser.add_argument("--data_dir", type=str, default="data")
  parser.add_argument("--cache_dir", type=str, default="train_cache")
  parser.add_argument("--output_path", type=str, default="runs", help="model save")
  parser.add_argument("--resume_from", type=str, default=None, help="resume training.")
  parser.add_argument("--first_cache_index", type=int, default=0)
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training (-1: not distributed)")
  parser.add_argument("--seed", type=int, default=43)
  parser.add_argument("--debug", action='store_true')
  args = parser.parse_args()

  args.output_path = os.path.join(args.output_path, f'pilot_{args.pilot_version}')
  args.cache_dir = os.path.join(args.data_dir, args.cache_dir)
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
  logger = setup_logger("trainer", distributed_rank=args.local_rank)

  # Model construction
  model = getattr(models, args.model)(args)
  model = model.to(device)
  optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay=args.wd)

  if args.loss_type == "MSELoss":
    criterion = nn.MSELoss(reduction='sum').to(device)
  else:
    criterion = getattr(nn, args.loss_type, getattr(auxiliary, args.loss_type, None))().to(device)
  criterion2 = nn.MSELoss(reduction='sum').to(device)

  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  train_dataset = SIGDataset(args, data_type="train")
  valid_dataset = SIGDataset(args, data_type="valid")
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
  valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
  train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True, shuffle=(not args.distributed))
  valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, pin_memory=True, shuffle=False)
  
  lr_scheduler = None
  if args.lr_scheduler == "linear":
    lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
  elif args.lr_scheduler == "cycle":
    lr_scheduler = LinearCyclicalScheduler(optimizer, 'lr', 0.0, args.lr, args.eval_iter * len(train_loader))
  elif args.lr_scheduler == "cosine":
    lr_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, args.eval_iter * len(train_loader))

  # Training function and trainer
  def update(engine, batch):
      model.train()
      y, x_label, y_pure, H = train_dataset.prepare_batch(batch, device=args.device)

      if args.with_pure_y and args.with_h:
        x_pred, y_pure_pred, H_pred = model(y, pure=y_pure, H=H, opp=True)
        loss_1 = criterion(x_pred, x_label) / args.gradient_accumulation_steps
        if args.loss_type == "MSELoss":
          loss_1 = loss_1 / x_pred.size(0)
        loss_noise = criterion2(y_pure_pred, y_pure) / y.size(0) / args.gradient_accumulation_steps
        loss_noise_h = criterion2(H_pred, H) / H.size(0) / args.gradient_accumulation_steps
        if args.only_l1:
          loss = loss_1
        else:
          loss = loss_1 + loss_noise * args.noise_lambda + loss_noise_h
        output = (loss.item(), loss_1.item(), loss_noise.item(), loss_noise_h.item())
      elif args.with_pure_y:
        x_pred, y_pure_pred = model(y, pure=y_pure if args.interpolation else None, opp=True)
        loss_1 = criterion(x_pred, x_label) / args.gradient_accumulation_steps
        loss_noise = criterion2(y_pure_pred, y_pure) / y.size(0) / args.gradient_accumulation_steps
        loss = loss_1 + loss_noise * args.noise_lambda
        output = (loss.item(), loss_1.item(), loss_noise.item())
      elif args.with_h:
        x_pred, H_pred = model(y, opp=True)
        loss_1 = criterion(x_pred, x_label) / args.gradient_accumulation_steps
        loss_noise = criterion2(H_pred, H) / H.size(0) / args.gradient_accumulation_steps
        loss = loss_1 + loss_noise * args.noise_lambda
        output = (loss.item(), loss_1.item(), loss_noise.item())
      else:
        x_pred = model(y)
        loss_1 = criterion(x_pred, x_label) / args.gradient_accumulation_steps
        loss = loss_1
        output = (loss.item(), loss_1.item(), torch.zeros_like(loss_1).item())

      loss.backward()
      if args.max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
      if engine.state.iteration % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
      return output
  trainer = Engine(update)

  to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
  metric_names = ["loss", "l1", "ln"]
  if args.with_pure_y and args.with_h:
    metric_names.append("lnH")

  common.setup_common_training_handlers(
    trainer=trainer,
    train_sampler=train_loader.sampler,
    to_save=to_save,
    save_every_iters=len(train_loader) * args.save_iter,
    lr_scheduler=lr_scheduler,
    output_names=metric_names,
    with_pbars=False,
    clear_cuda_cache=False,
    output_path=args.output_path,
    n_saved=2,
  )

  resume_from = args.resume_from
  if resume_from is not None:
    checkpoint_fp = Path(resume_from)
    assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
    logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    if args.reset_trainer:
      to_save.pop("trainer")
    checkpoint_to_load = to_save if 'validation' not in resume_from else {"model": model}
    Checkpoint.load_objects(to_load=checkpoint_to_load, checkpoint=checkpoint)
    if args.reset_lr_scheduler is not None:
      if args.reset_lr_scheduler == "linear":
        lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
      elif args.reset_lr_scheduler == "cycle":
        lr_scheduler = LinearCyclicalScheduler(optimizer, 'lr', 0.0, args.lr, args.eval_iter * len(train_loader))
      elif args.reset_lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 0.0, args.eval_iter * len(train_loader))

  metrics = {
    "accuracy": Accuracy(lambda output: (torch.round(output[0][0]), output[1][0])), 
    "loss_1": Loss(criterion, output_transform=lambda output: (output[0][0], output[1][0])),
    "loss_noise": Loss(criterion2, output_transform=lambda output: (output[0][1], output[1][1]))
  }
  if args.with_pure_y and args.with_h:
    metrics["loss_noise_h"] = Loss(criterion2, output_transform=lambda output: (output[0][2], output[1][2]))

  def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
    model.eval()
    with torch.no_grad():
      x, y, x_pure, H = valid_dataset.prepare_batch(batch, device=args.device, non_blocking=True)
      if args.with_pure_y and args.with_h:
        y_pred, x_pure_pred, h_pred = model(x, opp=True)
        outputs = (y_pred, x_pure_pred, h_pred), (y, x_pure, H)
      elif args.with_pure_y:
        y_pred, x_pure_pred = model(x, opp=True)
        outputs = (y_pred, x_pure_pred), (y, x_pure)
      elif args.with_h:
        y_pred, h_pred = model(x, opp=True)
        outputs = (y_pred, h_pred), (y, H)
      else:
        y_pred = model(x)
        x_pure_pred = x_pure
        outputs = (y_pred, x_pure_pred), (y, x_pure)       
      return outputs
  evaluator = Engine(_inference)
  for name, metric in metrics.items():
      metric.attach(evaluator, name)

  trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.eval_iter), lambda _: evaluator.run(valid_loader))

  if args.flush_dataset > 0:
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.n_epochs//args.flush_dataset), 
                  lambda _: train_loader.dataset.reset() if args.no_cache else train_loader.dataset.reload())

  # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
  if args.local_rank in [-1, 0]:
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=metric_names, output_transform=lambda _: {"lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

    tb_logger = common.setup_tb_logging(args.output_path, trainer, optimizer, evaluators={'validation': evaluator}, log_every_iters=1)

  # Store 3 best models by validation accuracy:
  common.gen_save_best_models_by_val_score(
    save_handler=DiskSaver(args.output_path, require_empty=False),
    evaluator=evaluator,
    models={"model": model},
    metric_name="accuracy",
    n_saved=3,
    trainer=trainer,
    tag="validation"
  )

  # Run the training
  trainer.run(train_loader, max_epochs=args.n_epochs)

  if args.local_rank in [-1, 0]:
    tb_logger.close()

if __name__ == '__main__':
  main()
