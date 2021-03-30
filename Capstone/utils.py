# utils.py - Utils for Python Code Gen project
import random
import torch
import yaml

def seed_code():
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def load_config(filename: str) -> dict:
    '''Load a configuration file as YAML'''
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    return config

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params:,} trainable parameters')
    return params

