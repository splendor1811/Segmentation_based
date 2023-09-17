import math

import torch.optim.lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


class Scheduler():
    def __init__(self, cfgs, optimizer: Optimizer):
        self.cfgs = cfgs
        self.optimizer = optimizer

    def get_scheduler(self) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        scheduler = CosineAnnealingLR(self.optimizer, self.cfgs['num_epochs'])
        return scheduler
