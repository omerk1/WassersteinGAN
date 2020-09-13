from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import Generator


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

    data_label_noise = (torch.rand(N, device=y_data.device)) * label_noise - label_noise / 2 + data_label
    generated_label_noise = (torch.rand(N, device=y_data.device)) * label_noise - label_noise / 2 + generated_label

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

    label_tensor = torch.ones(N, device=y_generated.device) * data_label
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
