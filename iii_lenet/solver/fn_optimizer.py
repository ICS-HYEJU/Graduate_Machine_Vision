import torch

def build_optimizer(cfg, model):
    if cfg['solver']['name'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg['solver']['lr0'],
                                    momentum=cfg['solver']['momentum'],
                                    nesterov=True
                                    )
    else:
        ############################################################################
        raise NotImplementedError('{} not implemented'.format(cfg['solver']['name']))
    return optimizer