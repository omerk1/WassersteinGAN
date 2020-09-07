from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======

        num_discriminator_features = 64
        in_channels = in_size[0]
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, num_discriminator_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_discriminator_features, num_discriminator_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_discriminator_features * 2, num_discriminator_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_discriminator_features * 4, num_discriminator_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_discriminator_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.convs(x)
        y = y.squeeze(1).squeeze(1).squeeze(1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======

        num_generated_features = 128
        self.de_conv = nn.Sequential(

            nn.ConvTranspose2d(z_dim, num_generated_features * 8, kernel_size=featuremap_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_generated_features * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features * 8, num_generated_features * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_generated_features * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features * 4, num_generated_features * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_generated_features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features * 2, num_generated_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_generated_features),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn(n, self.z_dim, device=device, requires_grad=with_grad)

        if with_grad:
            samples = self.forward(z)
        else:
            with torch.no_grad():
                samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.unsqueeze(2).unsqueeze(2)
        x = self.de_conv(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======

    generated_label = 1 - data_label
    N = y_data.shape[0]

    assert N == y_generated.shape[0]

    data_label_noise = (torch.rand(N, device=y_data.device))*label_noise - label_noise/2 + data_label
    generated_label_noise = (torch.rand(N, device=y_data.device))*label_noise - label_noise/2 + generated_label

    criterion = nn.BCEWithLogitsLoss()
    loss_data = criterion(y_data, data_label_noise)
    loss_generated = criterion(y_generated, generated_label_noise)

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    criterion = nn.BCEWithLogitsLoss()

    N = y_generated.shape[0]

    label_tensor = torch.ones(N, device=y_generated.device)*data_label
    loss = criterion(y_generated, label_tensor)

    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    generated_data = gen_model.sample(x_data.shape[0], with_grad=False)

    y_generated = dsc_model.forward(generated_data)

    y_data = dsc_model.forward(x_data)

    y_generated = y_generated.squeeze()
    y_data = y_data.squeeze()

    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_model.zero_grad()

    generated_data = gen_model.sample(x_data.shape[0], with_grad=True)
    y_generated = dsc_model.forward(generated_data)

    y_generated = y_generated.squeeze()

    gen_loss = gen_loss_fn(y_generated)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()
