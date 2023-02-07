import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same', bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', bias=False))
        self.bn1 = spectral_norm(nn.BatchNorm2d(128))
        self.conv3 = spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', bias=False))
        self.bn2 = spectral_norm(nn.BatchNorm2d(256))
        self.conv4 = spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', bias=False))
        self.bn3 = spectral_norm(512)
        self.conv5 = spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=0))
        
        self.apply(weights_init)

    def forward(self, input):
        out = nn.LeakyReLU(0.2, True)(self.conv1(input))
        out = self.conv2(out)
        out = nn.LeakyReLU(0.2, True)(self.bn1(out))
        out = self.conv3(out)
        out = nn.LeakyReLU(0.2, True)(self.bn2(out))
        out = self.conv4(out)
        out = nn.LeakyReLU(0.2, True)(self.bn3(out))
        out = self.conv5(out)
        
        out = out.view((input.shape[0], -1))
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, bias=False),
            nn.Tanh()
        )
        self.output_linear = nn.Linear(in_features=169, out_features=150)
        self.apply(weights_init)

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1)
        out = self.gen(input)
        out = out.view((-1, 169))
        out = self.output_linear(out)
        out = out.view((-1, 1, 10, 15))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)