# %% Imports
import os
import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18,resnet101

from src.utils.memory import log_mem, log_mem_amp, log_mem_amp_cp, log_mem_cp
from src.utils.plot import plot_mem, pp

class EncoderCompressed(nn.Module):
    def __init__(self, C=256):
        super(EncoderCompressed, self).__init__()
        m = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        layer2 = nn.Sequential(*list(m.backbone.layer2.children())[1:])

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=C, out_channels=C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=C, out_channels=C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=C, out_channels=C, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            layer2,
            m.backbone.layer3,
            m.backbone.layer4,
            m.classifier,
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )

    def forward(self, x, rate=None):
        return self.model(x)






base_dir = '.'
# %% Analysis baseline
os.environ['TORCH_HOME'] = os.path.join('../differentiabledata', '_logs',
                                        'models')
#model = resnet101().cuda()
#model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).cuda()

model = EncoderCompressed().cuda()
bs = 1
input_size = torch.rand(bs, 3, 512, 512).cuda()
model(input_size)

mem_log = []

try:
    mem_log.extend(log_mem(model, input_size, exp='baseline'))
except Exception as e:
    print(f'log_mem failed because of {e}')

df = pd.DataFrame(mem_log)

plot_mem(df, exps=['baseline'],
         output_file=f'{base_dir}/encoder_compressed_memory_plot_{bs}_{input_size.shape[2]}.png')

#pp(df, exp='baseline')

