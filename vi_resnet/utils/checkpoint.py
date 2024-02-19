import os
import torch

def load_networks(net, device, weight_path):
    assert os.path.exists(weight_path), f'There is no saved weight file...'
    print('loading the model from %s weight_path')
    state_dict = torch.load(weight_path, map_location=str(device))
    net.loade_state_dict(state_dict['model'])
    print('load completed....')
    return net