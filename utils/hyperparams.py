def gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )

    hypers['batch_size'] = 64

    hypers['z_dim'] = 100  # As illustrated in architecture in DCGAN paper
    hypers['data_label'] = 1  # Flip labels according to GAN Hacks https://github.com/soumith/ganhacks
    hypers['label_noise'] = 0.2  # Seems about right
    hypers['discriminator_optimizer'] = dict(
        type='Adam',
        lr=0.0002,
        betas=(0.5, 0.999)
    )

    hypers['generator_optimizer'] = dict(
        type='Adam',
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    return hypers


def wgan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )

    hypers['batch_size'] = 64

    hypers['z_dim'] = 100  # As illustrated in architecture in DCGAN paper
    hypers['data_label'] = 1  # Flip labels according to GAN Hacks https://github.com/soumith/ganhacks

    hypers['discriminator_optimizer'] = dict(
        type='RMSprop',
        lr=0.00005
    )

    hypers['generator_optimizer'] = dict(
        type='RMSprop',
        lr=0.00005
    )

    return hypers
