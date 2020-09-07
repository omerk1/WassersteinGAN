from data_loading import get_mnist_data_loader

# from models.gan import GAN
# from models.dcgan import DCGAN_MODEL
from ref.wgan_ref import WGAN


def main(args):
    model = None
    if args['model'] == 'GAN':
        # model = GAN(args)
        x = 6
    elif args['model'] == 'DCGAN':
        # model = DCGAN_MODEL(args)
        x = 6
    elif args['model'] == 'WGAN':
        model = WGAN(
            z_dim=args['z_dim'],
            img_channels=args['img_channels'],
            generator_iters=args['generator_iters']
        )
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader = get_mnist_data_loader(batch_size=args['batch_size'])
    # feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args['is_train']:
        model.train(train_loader)

    # start evaluating on test data
    else:
        pass
        # model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = {
        'model': 'WGAN',

        'z_dim': 100,
        'img_channels': 1,
        'generator_iters': 10,

        'batch_size': 64,
        'is_train': True,

    }
    main(args)
