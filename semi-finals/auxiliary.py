import torch
import os, datetime
import torch.nn as nn
import torch.nn.functional as F

def make_logdir(model_name: str):
  """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
  # Code copied from ignite repo
  current_time = datetime.now().strftime('%b%d_%H-%M-%S')
  logdir = os.path.join('runs', current_time + '_' + model_name)
  return logdir

def log_metrics(logger, epoch, elapsed, tag, metrics):
  logger.info(
      "\nEpoch {} - elapsed: {} - {} metrics:\n {}".format(
          epoch, elapsed, tag, "\n".join(["\t{}: {}".format(k, v) for k, v in metrics.items()])
      )
  )

class FocalLoss(nn.Module):
  def __init__(self, gamma=2, weight=None, ignore_index=-100):
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.weight = weight
    self.ignore_index=ignore_index

  def forward(self, input_, target):
    """
    input: [N, 1024]
    target: [N, 1024]
    """
    minus_input_ = 1 - input_
    loss = - target * (minus_input_)**self.gamma * torch.log(input_) - (1 - target) * (input_)**self.gamma * torch.log(minus_input_)
    return loss.mean()
