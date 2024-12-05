from torch import nn


class Discriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf,
    ):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(ndf, ndf * 2),
            self._block(ndf * 2, ndf * 4),
            self._block(ndf * 4, ndf * 8),
            nn.Conv2d(ndf * 8, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def _block(self, input_nc, output_nc):
        return nn.Sequential(
            nn.Conv2d(input_nc, output_nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_nc),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
