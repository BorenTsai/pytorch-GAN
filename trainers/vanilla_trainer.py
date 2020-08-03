import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from trainers.utils import *

class Trainer:
    def __init__(self,
                 generator,
                 discriminator,
                 dataloader,
                 k=1,
                 lr=1e-3,
                 num_epochs=100,
                 num_samples=10,
                 log_dir='experiments/VanillaGAN',
                 gpu_id=-1,
                 ):

        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.k = k
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.log_dir = log_dir
        self.gpu_id = gpu_id

        if (self.gpu_id not in [-1, None]) and (torch.cuda.is_available()):
            self.is_gpu = True
            torch.cuda.set_device(self.gpu_id)
            self.generator.to(self.gpu_id)
            self.discriminator.to(self.gpu_id)
        else:
            self.is_gpu = False
            self.gpu_id = None

        # Setting up logging directory
        make_dir(self.log_dir)

        print('\nInitializing Optimizers')
        self.G_optim = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.D_optim = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        print('\nDefining Criterion')
        self.criterion = torch.nn.BCELoss()

    def train(self):
        print('\nStaring Training')
        writer = SummaryWriter(self.log_dir)
        
        total_steps = 0

        for epoch in range(self.num_epochs):
            discriminator_loss = []
            generator_loss = []
            for data, _ in self.dataloader:

                if self.is_gpu:
                    data = data.to(self.gpu_id)

                D_batch_info = self.train_discriminator(data=data)
                G_batch_info = self.train_generator(data=data)
                
                discriminator_loss.append(D_batch_info['loss'])
                generator_loss.append(G_batch_info['loss'])

            mean_D_loss = np.asarray(discriminator_loss).mean()
            mean_G_loss = np.asarray(generator_loss).mean()

            print('Epoch: {}, Mean Discriminator Loss: {}, Mean Generator Loss: {}'.format(epoch, mean_D_loss, mean_G_loss))
            writer.add_scalar('Discriminator/AvgTotalLoss', mean_D_loss, epoch)
            writer.add_scalar('Generator/AvgLoss', mean_G_loss, epoch)

            if epoch % 10 == 0:
                self.write_images(writer, epoch=epoch)

        print('\nCompleted Training')
        self.write_images(writer, self.num_epochs)

    def train_discriminator(self, data):
        for step in range(self.k):
            self.D_optim.zero_grad()

            # create fake samples
            noise = torch.randn(data.size(0), self.generator.z_dim)
            if self.is_gpu:
                noise = noise.to(self.gpu_id)
            fake_samples = self.generator(noise)

            # get discriminator predictions
            D_real = self.discriminator(data)
            D_fake = self.discriminator(fake_samples)

            # compute loss and backprop
            ones = torch.ones(D_real.size(0), 1)
            zeros = torch.zeros(D_fake.size(0), 1)

            if self.is_gpu:
                ones = ones.to(self.gpu_id)
                zeros = zeros.to(self.gpu_id)

            real_loss = self.criterion(D_real, ones)
            fake_loss = self.criterion(D_fake, zeros)
            total_loss = real_loss + fake_loss

            total_loss.backward()
            self.D_optim.step()

        batch_info = {
            'loss': total_loss.data.item(),
            'real_loss': real_loss.data.item(),
            'fake_loss': fake_loss.data.item()
        }

        return batch_info

    def train_generator(self, data):
        self.G_optim.zero_grad()

        # create fake samples
        noise = torch.randn(data.size(0), self.generator.z_dim)
        if self.is_gpu:
            noise = noise.to(self.gpu_id)
        fake_samples = self.generator(noise)

        # test discriminator
        D_pred = self.discriminator(fake_samples)

        # compute loss and backprop
        ones = torch.ones(D_pred.size(0), 1)
        if self.is_gpu:
            ones = ones.to(self.gpu_id)
        loss = self.criterion(D_pred, ones)
        loss.backward()
        self.G_optim.step()

        batch_info = {
            'loss': loss.data.item()
        }
        return batch_info

    def write_images(self, writer, epoch):
        samples = self.generator.generate_samples(self.num_samples)
        for idx, sample in enumerate(samples):
            writer.add_image('Sample No. {}'.format(idx), sample, epoch)










