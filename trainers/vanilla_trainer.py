import torch
import torch.optim as optim

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
        print('\n Staring Training')
        writer = SummaryWriter(self.log_dir)

        for epoch in range(self.num_epochs):
            pbar = tqdm(total=len(self.dataloader.dataset))
            for data, _ in self.dataloader:

                if self.gpu_id:
                    data = data.to(self.gpu_id)

                D_batch_info = self.train_discriminator(data=data)

                writer.add_scalar('Discriminator/TotalLoss', D_batch_info['loss'])
                writer.add_scalar('Discriminator/RealLoss', D_batch_info['real_loss'])
                writer.add_scalar('Discriminator/FakeLoss', D_batch_info['fake_loss'])

                G_batch_info = self.train_generator(data=data)

                writer.add_scalar('Generator/Loss', D_batch_info['loss'])

                pbar.set_description('Epoch {}, Discriminator Loss {}, Generator Loss {}'.format(epoch, D_batch_info['loss'], G_batch_info['loss']))
                pbar.update(n=data.size(0))

            if epoch % 50 == 0:
                self.write_images(writer)

        print('\nCompleted Training')
        self.write_images(writer)

    def train_discriminator(self, data):
        for step in range(self.k):
            self.D_optim.zero_grad()

            # create fake samples
            noise = torch.randn(data.size(0), self.generator.z_dim)
            fake_samples = self.generator(noise)

            # get discriminator predictions
            D_real = self.discriminator(data)
            D_fake = self.discriminator(fake_samples)

            # compute loss and backprop
            real_loss = self.criterion(D_real, torch.zeros_like(D_real))
            fake_loss = self.criterion(D_fake, torch.ones_like(D_fake))
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
        fake_samples = self.generator(noise)

        # test discriminator
        D_pred = self.discriminator(fake_samples)

        # compute loss and backprop
        loss = self.criterion(D_pred, torch.ones_like(D_pred))
        loss.backward()
        self.G_optim.step()

        batch_info = {
            'loss': loss.data.item()
        }
        return batch_info

    def write_images(self, writer):
        samples = self.generator.generate_samples(self.num_samples)
        import ipdb; ipdb.set_trace()
        for idx, sample in enumerate(samples):
            writer.add_image('Sample No. {}'.format(idx), sample)










