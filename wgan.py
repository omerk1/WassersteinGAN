import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import Generator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader, dsc_iter_per_gen_iter: int = 5,
                weight_cliping_limit: float = 0.01):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    one = torch.tensor(1, dtype=torch.float)
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

        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        wasserstein_d = d_loss_real - d_loss_fake

        dsc_optimizer.step()

    for p in dsc_model.parameters():
        p.requires_grad = False

    for p in gen_model.parameters():
        p.requires_grad = True

    gen_model.zero_grad()

    fake_images = gen_model.sample(x_data.shape[0], with_grad=True)

    g_loss = dsc_model(fake_images)
    g_loss = g_loss.mean()
    g_loss.backward(mone)
    g_cost = -g_loss
    gen_optimizer.step()

    return d_loss.item(), wasserstein_d.item(), g_loss.item(), g_cost.item()
