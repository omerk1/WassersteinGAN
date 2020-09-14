import sys

import IPython.display
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





dsc_optimizer = create_optimizer(dsc.parameters(), hp['discriminator_optimizer'])
gen_optimizer = create_optimizer(gen.parameters(), hp['generator_optimizer'])

print(dsc_optimizer)
print(gen_optimizer)

labels = []
for i in range(len(data_set)):
    labels.append(data_set[i][1])
np.unique(labels)

# Training
num_epochs = 50

dsc_epoch_losses = []
w_epoch_losses = []
gen_epoch_losses = []
gen_epoch_costs = []

dsc_batch_losses = []
w_batch_losses = []
gen_batch_losses = []
gen_batch_costs = []

for epoch_idx in range(num_epochs):
    # We'll accumulate batch losses and show an average once per epoch.
    dsc_losses = []
    w_losses = []
    gen_losses = []
    gen_costs = []
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
            w_losses.append(wasserstein_d)
            gen_losses.append(gen_loss)
            gen_costs.append(gen_cost)

            dsc_batch_losses.append(dsc_loss)
            w_batch_losses.append(wasserstein_d)
            gen_batch_losses.append(gen_loss)
            gen_batch_costs.append(gen_cost)

            pbar.update()

    dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)
    w_avg_loss, gen_avg_cost = np.mean(w_losses), np.mean(gen_costs)

    dsc_epoch_losses.append(dsc_avg_loss)
    w_epoch_losses.append(w_avg_loss)
    gen_epoch_losses.append(gen_avg_loss)
    gen_epoch_costs.append(gen_avg_cost)

    samples = gen.sample(5, with_grad=False)
    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
    IPython.display.display(fig)
    plt.close(fig)

import scipy as sp
from scipy import signal

x = [i for i in range(len(w_batch_losses))]
plt.xlabel("generator iteration")
plt.ylabel("discriminator loss")
plt.plot(x, sp.signal.medfilt(w_batch_losses, 21))
plt.show()
plt.clf()
