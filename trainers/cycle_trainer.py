import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tensorboardX import SummaryWriter

from trainers.utils import *
from utils.image_buffer import ImageBuffer

class Trainer:
    def __init__(self,
                 G,
                 F,
                 Dx,
                 Dy,
                 train_dataloader,
                 test_dataloader,
                 lr=2e-4,
                 num_epochs=200,
                 gpu_id=None,
                 log_dir='./experiments/CycleGAN_test/',
                 img_buffer_capacity=50,
                 cycle_loss_lmbda=10):
        self.G = G
        self.F = F
        self.Dx = Dx
        self.Dy = Dy
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.lr = lr
        self.num_epochs = num_epochs
        self.gpu_id = gpu_id
        self.log_dir = log_dir
        self.img_buffer_capacity = img_buffer_capacity
        self.cycle_loss_lmbda = cycle_loss_lmbda

        self.fake_from_imgbuffer = ImageBuffer(max_capacity=self.img_buffer_capacity)
        self.fake_to_imgbuffer = ImageBuffer(max_capacity=self.img_buffer_capacity)

        self.ones = None
        self.zeros = None

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

        print('\nCreating Optimizers')
        gen_params  = list(self.G.parameters()) + list(self.F.parameters())
        disc_params = list(self.Dx.parameters()) + list(self.Dy.parameters())

        self.G_optim = optim.Adam(gen_params, lr=self.lr)
        self.D_optim = optim.Adam(disc_params, lr=self.lr)

        print('\nDefining Criterions')
        self.GANCriterion = nn.MSELoss()
        self.CycleCriterion = nn.L1Loss()

    def train(self):
        print('\nStaring Training')
        writer = SummaryWriter(self.log_dir)

        for epoch in range(self.num_epochs):
            pbar = tqdm(total=len(self.train_dataloader.dataset))
            G_losses = []
            D_losses = []

            for from_info, to_info in self.train_dataloader:
                G_loss, D_loss = self.forward(from_info, to_info, writer)
                G_losses.append(G_loss)
                D_losses.append(D_loss)

                pbar.set_description(
                    'Epoch {}, Avg Generator Loss {}, Avg Discriminator Loss {}'.format(epoch, np.mean(G_losses[-50:]),
                                                                                        np.mean(D_losses[-50:])))
                pbar.update(n=1)
                import ipdb; ipdb.set_trace()

            self.fake_from_imgbuffer.clear()
            self.fake_to_imgbuffer.clear()

            if epoch % 10 == 0:
                self.G.eval(); self.F.eval()
                self.Dx.eval(); self.Dy.eval()

                G_losses = []
                D_losses = []
                print('Epoch {}, Test Itr {}'.format(epoch // 10))
                for from_info, to_info in self.test_dataloader:
                    G_loss, D_loss = self.forward(from_info, to_info)
                    G_losses.append(G_loss); D_losses.append(D_loss)
                    
                G_mean_loss = np.mean(G_losses)
                D_mean_loss = np.mean(D_losses)

                print('Test Avg Generator Loss: {}'.format(G_mean_loss))
                print('Test Avg Discriminator Loss: {} '.format(D_mean_loss))

                self.write_images(writer=writer)

                self.fake_from_imgbuffer.clear()
                self.fake_to_imgbuffer.clear()


    def forward(self, from_info, to_info, writer=None):

        from_img = from_info['from_img']
        to_img = to_info['to_img']

        self.from_path = from_info['from_path']
        self.to_path = to_info['to_path']

        self.fake_to = self.G(from_img)
        self.fake_from = self.F(to_img)
        self.eval_fake_to = self.Dy(self.fake_to)
        self.eval_fake_from = self.Dx(self.fake_from)

        if self.ones is None and self.zeros is None:
            self.ones = torch.ones_like(self.eval_fake_from)
            self.zeros = torch.ones_like(self.eval_fake_from)

        self.toggle_discriminators(toggle=False)
        G_batch_info = self.train_generator(from_img, to_img)
        self.toggle_discriminators(toggle=True)
        D_batch_info = self.train_discriminator(from_img, to_img)

        if writer is not None:
            self.log_batch_info(G_batch_info=G_batch_info, D_batch_info=D_batch_info, writer=writer)

        return G_batch_info['loss'], D_batch_info['loss']

    def toggle_discriminators(self, toggle):
        for param_y, param_x in zip(self.Dy.parameters(), self.Dx.parameters()):
            param_y.requires_grad = toggle
            param_x.requires_grad = toggle

    def train_generator(self, from_img, to_img):
        self.G_optim.zero_grad()

        GANLoss = self.GANCriterion(self.eval_fake_from, torch.ones_like(self.eval_fake_from)) + \
                  self.GANCriterion(self.eval_fake_to, torch.ones_like(self.eval_fake_to))

        CycleLoss = self.CycleCriterion(from_img, self.F(self.fake_to)) + \
                       self.CycleCriterion(to_img, self.G(self.fake_from))
        CycleLoss *= self.cycle_loss_lmbda

        loss = GANLoss + CycleLoss
        loss.backward()
        self.G_optim.step()

        batch_info = {
            'loss': loss.item(),
            'GANLoss': GANLoss.item(),
            'CycleLoss': CycleLoss.item()
        }

        return batch_info

    def train_discriminator(self, from_img, to_img):
        self.D_optim.zero_grad()
        fake_to = self.fake_to_imgbuffer.get_img(self.fake_to, self.to_path)
        fake_from = self.fake_from_imgbuffer.get_img(self.fake_from, self.from_path)

        eval_fake_to = self.Dy(fake_to.detach())
        eval_fake_from = self.Dx(fake_from.detach())

        DyLoss = self.GANCriterion(self.Dy(to_img), self.ones) + \
                     self.GANCriterion(eval_fake_to, self.zeros)

        DxLoss = self.GANCriterion(self.Dx(from_img), self.ones) + \
                    self.GANCriterion(eval_fake_from, self.zeros)

        loss = 0.5 * (DyLoss + DxLoss)

        loss.backward()

        batch_info = {
            'loss': loss.item(),
            'DyLoss': DyLoss.item(),
            'DxLoss': DxLoss.item()
        }

        return batch_info

    def log_batch_info(self, G_batch_info, D_batch_info, writer):
        writer.add_scalar('Generator/Loss', G_batch_info['loss'])
        writer.add_scalar('Generator/GANLoss', G_batch_info['GANLoss'])
        writer.add_scalar('Generator/CycleLoss', G_batch_info['CycleLoss'])

        writer.add_scalar('Discriminator/TotalLoss', D_batch_info['loss'])
        writer.add_scalar('Discriminator/DyLoss', D_batch_info['DyLoss'])
        writer.add_scalar('Discriminator/DxLoss', D_batch_info['DxLoss'])
        
    def write_images(self, writer):
        G_samples = [self.fake_to_imgbuffer.sample() for _ in range(5)]
        F_samples = [self.fake_from_imgbuffer.sample() for _ in range(5)]

        for idx, sample in enumerate(samples):
            writer.add_image('Sample No. {}'.format(idx), sample)
        






