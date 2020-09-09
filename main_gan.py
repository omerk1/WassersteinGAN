import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import gan
import utils.plot as plot
from data_loading import get_mnist_dataset
from utils.hyperparams import gan_hyperparams

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def main():
    # Hyperparams
    hp = gan_hyperparams()
    batch_size = hp['batch_size']
    z_dim = hp['z_dim']

    # Data
    data_set = get_mnist_dataset()
    dl_train = DataLoader(data_set, batch_size, shuffle=True)
    im_size = data_set[0][0].shape

    # Model
    dsc = gan.Discriminator(im_size).to(device)
    gen = gan.Generator(z_dim, featuremap_size=4, out_channels=im_size[0]).to(device)

    # Optimizer
    def create_optimizer(model_params, opt_params):
        opt_params = opt_params.copy()
        optimizer_type = opt_params['type']
        opt_params.pop('type')
        return optim.__dict__[optimizer_type](model_params, **opt_params)

    dsc_optimizer = create_optimizer(dsc.parameters(), hp['discriminator_optimizer'])
    gen_optimizer = create_optimizer(gen.parameters(), hp['generator_optimizer'])

    # Loss
    def dsc_loss_fn(y_data, y_generated):
        return gan.discriminator_loss_fn(y_data, y_generated, hp['data_label'], hp['label_noise'])

    def gen_loss_fn(y_generated):
        return gan.generator_loss_fn(y_generated, hp['data_label'])

    # Training
    num_epochs = 50

    dsc_epoch_losses = []
    gen_epoch_losses = []

    dsc_batch_losses = []
    gen_batch_losses = []

    for epoch_idx in range(num_epochs):
        # We'll accumulate batch losses and show an average once per epoch.
        dsc_losses = []
        gen_losses = []
        print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

        with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
            for batch_idx, (x_data, _) in enumerate(dl_train):
                x_data = x_data.to(device)
                dsc_loss, gen_loss = gan.train_batch(
                    dsc, gen,
                    dsc_loss_fn, gen_loss_fn,
                    dsc_optimizer, gen_optimizer,
                    x_data)

                dsc_losses.append(dsc_loss)
                gen_losses.append(gen_loss)

                dsc_batch_losses.append(dsc_loss)
                gen_batch_losses.append(gen_loss)

                pbar.update()

        dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)

        dsc_epoch_losses.append(dsc_avg_loss)
        gen_epoch_losses.append(gen_avg_loss)

        print(f'Discriminator loss: {dsc_avg_loss}')
        print(f'Generator loss:     {gen_avg_loss}')

        samples = gen.sample(5, with_grad=False)
        fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
        # IPython.display.display(fig)
        plt.savefig('results/gan_reg/generated_samples_epoch_{}.png'.format(epoch_idx + 1))
        plt.close(fig)

    x = [i for i in range(len(dsc_epoch_losses))]
    plt.xlabel("epoch")
    plt.ylabel("discriminator loss")
    plt.plot(x, dsc_epoch_losses)
    plt.savefig('results/gan_reg/critic_loss_per_epoch.png')
    plt.clf()

    x = [i for i in range(len(dsc_batch_losses))]
    plt.xlabel("iteration")
    plt.ylabel("discriminator loss")
    plt.plot(x, dsc_batch_losses)
    plt.savefig('results/gan_reg/critic_loss_per_iteration.png')
    plt.clf()


if __name__ == '__main__':
    main()
