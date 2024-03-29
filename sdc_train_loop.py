# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tq

#import torch.distributions as D
import torch.nn.functional as Ff

import torchvision

import numpy as np
from pprint import pprint
import math

from scenetransformer import SceneTransformer
from sdcdataset import SceneDataset

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ysdc_dataset_api.evaluation.metrics import (
    average_displacement_error_torch, final_displacement_error_torch,
    batch_mean_metric_torch)

def lnloss(out, out_prob, gt, mask):
  F = out.shape[1]
  A = out.shape[2]
  T = out.shape[3]
  gt = gt.unsqueeze(1).repeat(1, F, 1, 1, 1)

  term0 = torch.log(torch.abs(out_prob))

  term1 = torch.sum(torch.log(torch.abs(out[:, :, :, :, 2])/2.), axis = 3)
  term2 = torch.sum(torch.log(torch.abs(out[:, :, :, :, 3])/2.), axis = 3)

  term3 = torch.sum(torch.abs(out[:, :, :, :, 2])*torch.abs(out[:, :, :, :, 0] - gt[:, :, :, :, 0]), axis = 3)
  term4 = torch.sum(torch.abs(out[:, :, :, :, 3])*torch.abs(out[:, :, :, :, 1] - gt[:, :, :, :, 1]), axis = 3)

  exp = term0+term1+term2-term3-term4
  #exp = term0 + term1 - term3 - term4

  loss = -torch.sum(mask*torch.logsumexp(exp, 1), 1)/mask.sum(axis = 1)
  return loss

def train_step(
    batch,
    clip: bool = True,
    **kwargs
):
    """Performs a single gradient-descent optimization step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    x, gt, mask = batch
    x = x.to(device)
    gt = gt.to(device)
    mask = mask.to(device)

    # Forward pass from the model.
    # Stores the contextual encoding in model._z
    out, p = model(x)

    # Calculates loss (NLL).
    loss = lnloss(out[:, :, :, 25:, :], p, gt, mask).sum()
    # if loss == tensor(nan)
    # Backward pass.
    loss.backward()

    # Clips gradients norm.

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    # Performs a gradient descent step.
    if not torch.isnan(loss):
      optimizer.step()
      scheduler.step()
      print(optimizer.param_groups[0]["lr"])

    # # Compute additional metrics.
    # ade = batch_mean_metric_torch(
    #     base_metric=average_displacement_error_torch,
    #     predictions=predictions,
    #     ground_truth=y)
    # fde = batch_mean_metric_torch(
    #     base_metric=final_displacement_error_torch,
    #     predictions=predictions,
    #     ground_truth=y)
    # loss_dict = {
    #     'nll': loss.detach(),
    #     'ade': ade.detach(),
    #     'fde': fde.detach()}

    return loss.detach()

#------SPACE OF PARAMETERS---------------
dataset_path = '/data/train/'
scene_tags_fpath = '/data/tags.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A = 100
F = 5
D = 128
B = 3
T = 50
in_features = 8
weight_decay = 0.0
lr = 1e-4
exp_name = f'D={D} B={B} test'


num_epochs = 1
num_warmup_epochs = 0.1
clip_gradients = 5.0
num_workers = 2
checkpoint_dir = '/checkpoints/'

epoch = 1
#-----------------------

dataset = SceneDataset(
    dataset_path=dataset_path,
    scene_tags_fpath=scene_tags_fpath,
    feature_producer=None,
    transform_ground_truth_to_agent_frame=True,
    A = A
)

train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=B,
            num_workers=num_workers)

model = SceneTransformer(in_features, F, A, T, D).to(device)

optimizer = optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=round(0.1*round(388406/B)),
    num_training_steps=round(388406/B))





def train_epoch(
      dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
  """
  Performs an epoch of gradient descent optimization on `dataloader`."""
  model.train()
  #train_loss = []
  steps = 0
  with tq.tqdm(dataloader) as pbar:
      for batch in pbar:

          # Performs a gradient-descent step.
          loss_b = train_step(batch)

          train_loss.append(loss_b.item())

          steps += 1
          if steps % 3000 == 0:
            pth = checkpoint_dir + f'{epoch}_{steps}.pth'
            torch.save(model.state_dict(), pth)


  return train_loss #, steps

train_loss = []
train_epoch(train_dataloader)
