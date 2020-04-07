# %% Imports
import os
import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18,resnet101

from src.utils.memory import log_mem, log_mem_amp, log_mem_amp_cp, log_mem_cp
from src.utils.plot import plot_mem, pp

base_dir = '.'
# %% Analysis baseline
os.environ['TORCH_HOME'] = os.path.join('../differentiabledata', '_logs',
                                        'models')
model = resnet101().cuda()
#model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False).cuda()

bs = 1
input_size = torch.rand(bs, 3, 512, 512).cuda()

mem_log = []

try:
    mem_log.extend(log_mem(model, input_size, exp='baseline'))
except Exception as e:
    print(f'log_mem failed because of {e}')

df = pd.DataFrame(mem_log)

plot_mem(df, exps=['baseline'],
         output_file=f'{base_dir}/resnet101_memory_plot_{bs}_{input_size.shape[2]}.png')

#pp(df, exp='baseline')

