import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_size, model_type):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size

        self.num_discriminator_features = 64
        self.model_type = model_type
        in_channels = in_size[0]
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, self.num_discriminator_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_discriminator_features, self.num_discriminator_features * 2, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.num_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_discriminator_features * 2, self.num_discriminator_features * 4, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.num_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_discriminator_features * 4, self.num_discriminator_features * 8, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.num_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if self.model_type == 'DCGAN':
            self.output = nn.Conv2d(self.num_discriminator_features * 8, 1, kernel_size=4, stride=1,
                                    padding=0, bias=False)
        elif self.model_type == 'WGAN':
            self.output = nn.Linear(8 * 16 * self.num_discriminator_features, 1)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """

        y = self.convs(x)
        if self.model_type == 'DCGAN':
            y = self.output(y)
            y = y.squeeze(1).squeeze(1).squeeze(1)

        elif self.model_type == 'WGAN':
            y = self.output(y.view(-1, 8 * 16 * self.num_discriminator_features))
            y = y.squeeze(1)
        return y
