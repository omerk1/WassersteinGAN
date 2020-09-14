import numpy as np
import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Generator(nn.Module):
    def __init__(self, z_dim, img_shape, featuremap_size=4, generator_type='DCGAN'):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        self.img_shape = img_shape
        self.out_channels = img_shape[0]
        num_generated_features = 128
        self.generator_type = generator_type
        if self.generator_type == 'DCGAN':
            self.gen = nn.Sequential(

                nn.ConvTranspose2d(z_dim, num_generated_features * 8, kernel_size=featuremap_size, stride=1,
                                   padding=0, bias=False),
                nn.BatchNorm2d(num_generated_features * 8),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(num_generated_features * 8, num_generated_features * 4, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(num_generated_features * 4),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(num_generated_features * 4, num_generated_features * 2, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(num_generated_features * 2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(num_generated_features * 2, num_generated_features, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(num_generated_features),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(num_generated_features, self.out_channels, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                nn.Tanh()
            )
        elif generator_type == 'MLP':
            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                # layers.append(nn.ReLU())
                return layers
            bn = False
            self.gen = nn.Sequential(
                *block(z_dim, 512, normalize=bn),
                *block(512, 512, normalize=bn),
                *block(512, 512, normalize=bn),
                *block(512, 512),
                nn.Linear(512, int(np.prod(img_shape))),
                nn.Tanh()
            )

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """

        z = torch.randn(n, self.z_dim, device=device, requires_grad=with_grad)

        if with_grad:
            samples = self.forward(z)
        else:
            with torch.no_grad():
                samples = self.forward(z)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """

        if self.generator_type == 'DCGAN':
            z = z.unsqueeze(2).unsqueeze(2)
        x = self.gen(z)

        if self.generator_type == 'MLP':
            x = x.reshape(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])

        return x
