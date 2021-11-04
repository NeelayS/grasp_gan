import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GraspingDataset
from loss import GANLoss
from model import define_D, define_G


data_dir = "/content/drive/My Drive/Grasping GAN/processed"
model_dir = "/content/drive/My Drive/Grasping GAN/models"
batch_size = 8
epochs = 1
lr = 0.01

dataset = GraspingDataset(data_dir)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_g = define_G(3, 3, 64, "batch", False, "normal", 0.02, gpu_id=device)
net_d = define_D(3 + 3, 64, "basic", gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

optimizer_g = optim.Adam(net_g.parameters(), lr=lr)
optimizer_d = optim.Adam(net_d.parameters(), lr=lr)

l1_weight = 10

for epoch in range(epochs):
    # train
    for iteration, batch in enumerate(data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * l1_weight

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()

        optimizer_g.step()

        print(
            "===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(data_loader), loss_d.item(), loss_g.item()
            )
        )

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    torch.save(net_g.state_dict(), os.path.join(model_dir, "generator.pth"))
    torch.save(net_d.state_dict(), os.path.join(model_dir, "discriminator.pth"))
