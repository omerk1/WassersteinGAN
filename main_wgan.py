import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import discriminator
import generator
import utils.plot as plot
import wgan
from data_loading import get_mnist_dataset
from utils.hyperparams import wgan_hyperparams

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def main():
    # Hyperparams
    hp = wgan_hyperparams()
    batch_size = hp['batch_size']
    z_dim = hp['z_dim']

    # Data
    data_set = get_mnist_dataset()
    dl_train = DataLoader(data_set, batch_size, shuffle=True)
    im_size = data_set[0][0].shape

    # Model
    dsc = discriminator.Discriminator(im_size, model_type='WGAN').to(device)
    gen = generator.Generator(z_dim, img_shape=im_size, featuremap_size=4, generator_type='MLP').to(device)

    # Optimizer
    def create_optimizer(model_params, opt_params):
        opt_params_tmp = opt_params.copy()
        optimizer_type = opt_params_tmp.pop('type')
        return optim.__dict__[optimizer_type](model_params, **opt_params_tmp)

    dsc_optimizer = create_optimizer(dsc.parameters(), hp['discriminator_optimizer'])
    gen_optimizer = create_optimizer(gen.parameters(), hp['generator_optimizer'])

    # Training
    num_epochs = 10
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
                dsc_loss, wasserstein_d, gen_loss, gen_cost = wgan.train_batch(
                    dsc_model=dsc, gen_model=gen,
                    dsc_optimizer=dsc_optimizer, gen_optimizer=gen_optimizer,
                    x_data=x_data, dsc_iter_per_gen_iter=5, weight_cliping_limit=0.01
                )

                dsc_losses.append(dsc_loss)
                gen_losses.append(gen_loss)

                dsc_batch_losses.append(dsc_loss)
                gen_batch_losses.append(gen_loss)

                pbar.update()

        dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)
        dsc_epoch_losses.append(dsc_avg_loss)
        gen_epoch_losses.append(gen_avg_loss)

        samples = gen.sample(5, with_grad=False)
        fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
        # IPython.display.display(fig)
        plt.savefig('results/wgan_reg/generated_samples_epoch_{}.png'.format(epoch_idx + 1))
        plt.close(fig)

        print(f'Discriminator loss: {dsc_avg_loss}')
        print(f'Generator loss:     {gen_avg_loss}')

    x = [i for i in range(len(dsc_epoch_losses))]
    plt.xlabel("epoch")
    plt.ylabel("discriminator loss")
    plt.plot(x, dsc_epoch_losses)
    plt.savefig('results/wgan_reg/critic_loss_per_epoch.png')
    plt.clf()

    x = [i for i in range(len(dsc_batch_losses))]
    plt.xlabel("iteration")
    plt.ylabel("discriminator loss")
    plt.plot(x, dsc_batch_losses)
    plt.savefig('results/wgan_reg/critic_loss_per_iteration.png')
    plt.clf()


if __name__ == '__main__':
    main()
