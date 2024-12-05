from torch import nn


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(input_nc, ngf * 16, 4, 1, 0),
            self._block(ngf * 16, ngf * 8, 4, 2, 1),
            self._block(ngf * 8, ngf * 4, 4, 2, 1),
            self._block(ngf * 4, ngf * 2, 4, 2, 1),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)
