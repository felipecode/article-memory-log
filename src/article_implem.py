# %% Imports
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

from src.utils.memory import log_mem, log_mem_amp, log_mem_amp_cp, log_mem_cp
from src.utils.plot import plot_mem, pp

base_dir = '.'
# %% Analysis baseline

model = resnet18().cuda()
bs = 128
input = torch.rand(bs, 3, 224, 224).cuda()

mem_log = []

try:
    mem_log.extend(log_mem(model, input, exp='baseline'))
except Exception as e:
    print(f'log_mem failed because of {e}')

df = pd.DataFrame(mem_log)

plot_mem(df, exps=['baseline'], output_file=f'{base_dir}/baseline_memory_plot_{bs}.png')

#pp(df, exp='baseline')

