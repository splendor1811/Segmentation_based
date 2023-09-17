import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from datasets.dataloader import get_loader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import yaml
import wandb
import numpy as np
import random
import csv
import glob
import json
import shutil
import traceback
from collections import OrderedDict
import pickle
import time
import copy
import gc
from datasets.dataloader import TrainDataset, EvalDataset

from models.clone_model import Meta_Polypv2
from metrics.schedulers.schedulers import Scheduler
from metrics.optimizers.optimizers import Optimizer
from metrics.losses.losses import MetaPolypv2_Loss
from metrics.metrics import Dice_Coeff, IoU


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)