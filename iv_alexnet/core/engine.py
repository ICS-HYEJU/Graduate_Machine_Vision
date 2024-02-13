from torch.utils.tensorboard import SummaryWriter
from ..utils.events import write_tbimg, write_tbloss, write_tbacc, write_tbPR
from tqdm import tqdm
import time
import os
import torch
import numpy as np

from ..dataset import create_dataloader

class Trainer():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device

        # ===== save path =====
        self.save_path = self.make_save_path()

        # ===== tensorboard =====
        self.tblogger = SummaryWriter(self.save_path)

        # ===== DataLoader =====
        self.train_loader, self.val_loader = self.get_dataloader()

        # ===== Model =====
        self.model = self.build_model()

        # ===== Optimiaer =====
        self.optimizer = self.build_optimizer()

        # ===== Scheduler =====

        # ===== Loss =====
        self.criterion = self.set_criterion()

        # ===== Parameters =====
        self.max_epoch = self.cfg['solver']['save_base_path']
        self.max_stepnum = len(self.train_loader)

    def calc_loss(self, logits, labels):
        return self.criterion(logits, labels.float())

    def set_criterion(self):
        return torch.nn.BCEWithLogitsLoss(reduction='sum').to(self.device)

    def build_optimizer(self):
        from ..solver.fn_optimizer import build_optimizer
        optim = build_optimizer(self.cfg, self.model)

    def build_model(self):
        name = self.cfg['model']['name']
        if name == 'lenet':
            from ..model.lenet import LeNet
            model = LeNet().to(self.device)
        elif name == 'alexnet':
            from ..model.alexnet import alexnet
            model = alexnet().to(self.device)
        else:
            raise NotImplementedError(f'The required model is not implemented yet...')
        return model

    def get_dataloader(self):
        train_loader, val_loader = create_dataloader(self.cfg['dataset'])

    def make_save_path(self):
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name'])
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def start_train(self):
        try:
            print(f'Training start...')
            start_time = time.time()
            for epoch in range(self.max_epoch):
                self.train_one_epoch(epoch)
            print(f'\nTraining completed in {(time.time() - start_time) / 3600:.3f} hours.')
        except Exception as _:
            print('ERROR in training loop or eval/save model.')
            raise

    def train_one_epoch(self, epoch):
        try:
            TP = np.zeros(8)
            FP = np.zeros(8)
            FN = np.zeros(8)

            pbar = tqdm(enumerate(self.train_loader), total=self.max_stepnum)
            for step, self.batch_data in pbar:
                imgs = self.batch_data[0].to(self.device)
                labels = self.batch_data[1].to(self.device)

                # Run model
                logits = self.model(imgs)

                # Calculate Loss
                loss = self.calc_loss(logits, labels)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Get statistics
                TP, FP, FN = self.get_statistics(self.model.predict(logits.detatch()), labels,
                                                 TP, FP, FN)

                # Print input images
                if step % 2 == 0:
                    # write_tbimg(self.tblogger, imgs, (epoch * self.max_stepnum) + step)
                    # Logging Loss
                    write_tbimg(self.tblogger, loss.detatch().cpu(), (epoch * self.max_stepnum) + step)

                # Logging Acc
                # TP / (TP + FP) : actually True for model predict = True
                write_tbimg(self.tblogger, TP / (TP + FP), epoch, 'train')
                write_tbPR(self.tblogger, TP, FP, epoch, 'train')

        except Exception as _:
            print('ERROR in training steps')
            raise

    @staticmethod
    def get_statistics(p, t, TP, FP, FN):
        # TP : Ture Positive
        # FP : False Positive
        # FN : False Negative
        # High Accuracy = High TP, FP
        for defect_idx in range(p.shape[1]):
            p_per_defect = p[:, defect_idx].cpu().detach().numpy()
            t_per_defect = t[:, defect_idx].cpu().detach().numpy()

            TP[defect_idx] += np.sum(p_per_defect * t_per_defect)
            FP[defect_idx] += np.sum(p_per_defect * (1 - t_per_defect))
            FN[defect_idx] += np.sum((1 - p_per_defect) * t_per_defect)
        return TP, FP, FN