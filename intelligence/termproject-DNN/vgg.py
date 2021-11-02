import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VGG16(nn.Module):

    def __init__(self, _input_channel, num_class):
        super().__init__()
        # 모델 구현
        self.fc_in_features = 512 * 1 * 1
        self.vgg_func = nn.Sequential(
                        nn.Conv2d(in_channels=_input_channel, out_channels=64, kernel_size = 3, padding = 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3, padding = 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2), # kernel size, stride

                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),

                        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),

                        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),

                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Flatten(), #x = x.view(-1, self.fc_in_features)

                        nn.Linear(in_features=self.fc_in_features, out_features=256),
                        nn.Linear(in_features=256, out_features=64),
                        nn.Linear(in_features=64, out_features=num_class),
                        nn.Softmax(dim=-1)
                    )
    

    def forward(self, x):
        # forward 구현

        return self.vgg_func(x)