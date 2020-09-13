import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


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

        self.num_discriminator_features = 64
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

            # nn.Conv2d(self.num_discriminator_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

        self.output = nn.Linear(8 * 16 * self.num_discriminator_features, 1)
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
        y = self.output(y.view(-1, 8 * 16 * self.num_discriminator_features))
        y = y.squeeze(1)
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

            nn.ConvTranspose2d(z_dim, num_generated_features * 8, kernel_size=featuremap_size, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_generated_features * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features * 8, num_generated_features * 4, kernel_size=4, stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_generated_features * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features * 4, num_generated_features * 2, kernel_size=4, stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_generated_features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features * 2, num_generated_features, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_generated_features),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_generated_features, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.output = nn.Tanh()
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
        # device = next(self.parameters()).device
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
        x = self.output(x)
        # ========================
        return x


def calc_gradient_penalty(netD, real_data, fake_data, bs=64, lambda_val=10):
    alpha = torch.rand(bs, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_val
    return gradient_penalty


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader, dsc_iter_per_gen_iter: int = 5,
                weight_cliping_limit: float = 0.01):
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
    one = torch.tensor(1, dtype=torch.float)
    # one = torch.FloatTensor([1])
    mone = one * -1

    one = one.to(device)
    mone = mone.to(device)

    d_loss = 0
    wasserstein_d = 0

    for p in gen_model.parameters():
        p.requires_grad = False

    for p in dsc_model.parameters():
        p.requires_grad = True

    for _ in range(dsc_iter_per_gen_iter):
        dsc_optimizer.zero_grad()

        for p in dsc_model.parameters():
            p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

        d_loss_real = dsc_model(x_data)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone)

        fake_images = gen_model.sample(x_data.shape[0], with_grad=False)
        inputv = fake_images
        d_loss_fake = dsc_model(inputv)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)

        gradient_penalty = 0

        # gradient_penalty = calc_gradient_penalty(dsc_model, x_data.data, fake_images.data)
        # gradient_penalty.backward()

        # d_loss += d_loss_fake - d_loss_real + gradient_penalty
        # wasserstein_d += d_loss_real - d_loss_fake

        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        wasserstein_d = d_loss_real - d_loss_fake

        dsc_optimizer.step()

    # d_loss = d_loss / dsc_iter_per_gen_iter
    # wasserstein_d = wasserstein_d / dsc_iter_per_gen_iter
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    for p in dsc_model.parameters():
        p.requires_grad = False

    for p in gen_model.parameters():
        p.requires_grad = True

    gen_model.zero_grad()

    fake_images = gen_model.sample(x_data.shape[0], with_grad=True)

    g_loss = dsc_model(fake_images)
    # g_loss = g_loss.mean().mean(0).view(1)
    g_loss = g_loss.mean()  # .mean().mean()
    g_loss.backward(mone)
    g_cost = -g_loss
    gen_optimizer.step()
    # ========================

    return d_loss.item(), wasserstein_d.item(), g_loss.item(), g_cost.item()
