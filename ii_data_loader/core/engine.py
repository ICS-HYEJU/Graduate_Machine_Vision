from torch.utils.tensorboard import SummaryWriter
from ..utils.events import write_tbimg
from tqdm import tqdm
import time
import os
import torch

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

        # ===== Optimiaer =====

        # ===== Scheduler =====

        # ===== Loss =====

        # ===== Parameters =====
        self.max_epoch = self.cfg['solver']['save_base_path']
        self.max_stepnum = len(self.train_loader)

    def get_dataloader(self):
        train_loader, val_loader = create_dataloader(self.cfg['dataset'])
        return train_loader, val_loader

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
            pbar = tqdm(enumerate(self.train_loader), total=self.max_stepnum)
            for step, self.batch_data in pbar:
                # SHOULD IMPLLEMENT THE PART OF TRAINING A MODEL
                imgs = self.batch_data[0]
                labels = self.batch_data[1]

                # Print input images
                if step % 1 == 0:
                    write_tbimg(self.tblogger, imgs, (epoch * self.max_stepnum) + step)

        except Exception as _:
            print('ERROR in training steps')
            raise
