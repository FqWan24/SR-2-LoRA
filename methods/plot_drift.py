import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy, count_parameters
from models.sinet_lora import SiNet
from models.vit_lora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import optimgrad
import re
from collections import defaultdict
from utils.losses import AugmentedTripletLoss
from scipy.spatial.distance import cdist

import sklearn
import os
import random
import matplotlib.pyplot as plt


class Drift(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self.cls_mean = []
        self.cls_cov = []
        self.cls_mean_shift = []


    def incremental_eval_draw_shift(self):
        # recore first time feature
        self._compute_mean()
        
        if self._cur_task == 0:
            return

        current_task_shift_list = []

        with torch.no_grad():
            for class_idx in range(self._known_classes):
                task_id = class_idx//(self._total_classes-self._known_classes)
                data, targets, idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4
                )

                vectors = []
                for _, _inputs, _targets in idx_loader:
                    inputs, targets = _inputs.to(self._device), _targets.to(self._device)
                    _vectors = self._network.extract_vector(inputs)
                    # _vectors = self._network.extract_vector_by_noise(inputs,task_id,noise_id=task_id)
                    vectors.append(_vectors)
                vectors = torch.cat(vectors, dim=0)
                current_mean  = vectors.mean(dim=0).to(self._device)

                initial_mean = self.cls_mean[class_idx].to(self._device)

                norm_current = F.normalize(current_mean.unsqueeze(0), p=2, dim=1).squeeze(0)
                norm_initial = F.normalize(initial_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

                dist = torch.norm(norm_current  - norm_initial, p=2).item()

                current_task_shift_list.append(dist)
        
        self.cls_mean_shift.append(current_task_shift_list)
        self.plot_prototype_shift()

    def plot_prototype_shift(self):

        filename = 'task_'+str(self._total_classes//(self._total_classes-self._known_classes))+'.png'
        save_path = os.path.join(self.args['log_dir'], filename) # 假设你有 log_dir

        # self.cls_mean_shift 是你的两层list
        tasks = range(1, len(self.cls_mean_shift) + 2)
        avg_shifts = []
        avg_shifts.append(0.0)

        for t_idx, shift_list in enumerate(self.cls_mean_shift):
        
            old_class_shifts = [d for d in shift_list]
            
            avg_dist = sum(old_class_shifts) / len(old_class_shifts)
                
            avg_shifts.append(avg_dist)

        # 绘图
        plt.figure(figsize=(8, 5))
        plt.plot(tasks, avg_shifts, marker='o', label='Ours')
        plt.xlabel('Task')
        plt.ylabel('Distance (Prototype Shift)')
        plt.title('Prototype Shift across Tasks')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)

    @torch.no_grad()
    def _compute_mean(self):
        self._network.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                inputs, targets = _inputs.to(self._device), _targets.to(self._device)
                _vectors = self._network.extract_vector(inputs)
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            
            features_per_cls = vectors

            self.cls_mean.append(features_per_cls.mean(dim=0).to(self._device))
            self.cls_cov.append(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-2).to(self._device))