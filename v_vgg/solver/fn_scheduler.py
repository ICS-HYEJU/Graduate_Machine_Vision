import torch

def build_scheduler(cfg, optimizer):
    if cfg['scheduler']['name'] == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    elif cfg['scheduler']['name'] == 'cycliclr':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=cfg['scheduler']['lr_min'],
                                                      max_lr=cfg['scheduler']['lr_max'],
                                                      cycle_momentum=False,
                                                      step_size_up=5,
                                                      step_size_down=8,
                                                      mode='triangular2')
    elif cfg['scheduler']['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=10,
                                                               eta_min=cfg['scheduler']['lr_min'])
    else:
        raise NotImplementedError('{} not implemented'.format(cfg['scheduler']['name']))
    return scheduler