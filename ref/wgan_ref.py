import os

import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Generator(torch.nn.Module):
    def __init__(self, z_dim, out_channels):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=out_channels, kernel_size=4, stride=2, padding=1))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.gen(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.disc = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=0))

    def forward(self, x):
        x = self.disc(x)
        return self.output(x).squeeze(2).squeeze(2)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.disc(x)
        return x.view(-1, 1024 * 4 * 4)


class WGAN(object):
    def __init__(self, z_dim, img_channels, generator_iters):
        self.gen = Generator(z_dim=z_dim, out_channels=img_channels).to(device)
        self.disc = Discriminator(in_channels=img_channels).to(device)

        self.img_channels = img_channels

        # WGAN values from paper
        self.learning_rate = 0.00005

        self.batch_size = 64
        self.weight_cliping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = torch.optim.RMSprop(self.disc.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=self.learning_rate)

        # Set the logger
        # self.logger = Logger('./logs')
        # self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = generator_iters
        self.critic_iter = 5

    # def check_cuda(self, cuda_flag=False):
    #     if cuda_flag:
    #         self.cuda_index = 0
    #         self.cuda = True
    #         self.D.cuda()
    #         self.G.cuda()
    #         print("Cuda enabled flag: {}".format(self.cuda))

    def train(self, train_loader):
        # self.t_begin = t.time()
        # self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.images_batches_generator(train_loader)

        pos_label = torch.FloatTensor([1]).to(device)
        neg_label = (pos_label * -1).to(device)

        for g_iter in range(self.generator_iters):

            # Requires grad, Generator requires_grad = False
            for p in self.disc.parameters():
                p.requires_grad = True

            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.disc.zero_grad()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                for p in self.disc.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                images = self.data.__next__().to(device)
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.disc(images)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(pos_label)

                # Train with fake images
                z = torch.randn(self.batch_size, 100, 1, 1).to(device)

                fake_images = self.gen(z)
                d_loss_fake = self.disc(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(neg_label)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

            # Generator update
            for p in self.disc.parameters():
                p.requires_grad = False  # to avoid computation

            self.gen.zero_grad()

            # Train generator
            # Compute loss with fake images
            z = torch.randn(self.batch_size, 100, 1, 1).to(device)
            fake_images = self.gen(z)
            g_loss = self.disc(fake_images)
            g_loss = g_loss.mean().mean(0).view(1)
            g_loss.backward(pos_label)
            g_cost = -g_loss
            self.g_optimizer.step()

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % 1000 == 0:
                self.save_model()
                # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(10):
                #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                #
                # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')

                # Denormalize images and save them in grid 8x8
                z = torch.randn(800, 100, 1, 1).to(device)
                samples = self.gen(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                # grid = utils.make_grid(samples)
                # utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                # Testing
                # time = t.time() - self.t_begin
                # print("Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                # print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                # output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                # self.file.write(output)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Wasserstein distance': Wasserstein_D.data[0],
                    'Loss D': d_loss.data[0],
                    'Loss G': g_cost.data[0],
                    'Loss D Real': d_loss_real.data[0],
                    'Loss D Fake': d_loss_fake.data[0]

                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)

        # self.t_end = t.time()
        # print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # self.file.close()

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = torch.randn(self.batch_size, 100, 1, 1).to(device)
        samples = self.gen(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        # grid = utils.make_grid(samples)
        # print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        # utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if self.img_channels == 3:
            return self.to_np(images.view(-1, self.img_channels, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.gen(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.img_channels == 3:
                generated_images.append(sample.reshape(self.img_channels, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.gen.state_dict(), './generator.pkl')
        torch.save(self.disc.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.disc.load_state_dict(torch.load(D_model_path))
        self.gen.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def images_batches_generator(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1).to(device)
        z1 = torch.randn(1, 100, 1, 1).to(device)
        z2 = torch.randn(1, 100, 1, 1).to(device)

        # z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.gen(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.img_channels, 32, 32).data.cpu())

        # grid = utils.make_grid(images, nrow=number_int)
        # utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        # print("Saved interpolated images.")
