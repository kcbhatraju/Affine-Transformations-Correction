import torch, torchviz
from torch import nn, optim, autograd
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import numpy as np
from diffaff import affine

import warnings
warnings.filterwarnings("ignore")

def plot_image_grid(images, ncols=None, cmap="gray"):
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()): 
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)
    plt.show()

def progress(current,total,length=25,**kwargs):
    done_token, current_token = ("=", ">")
    token_arr = []
    bar = total // length
    token_arr.extend([done_token]*(current//bar))
    if (total-current): token_arr.extend([current_token])
    attrs = json.dumps(kwargs).replace('"',"")[1:-1]
    final = f"{current}/{total} [{''.join(token_arr)}{' '*max(0,length-current//bar)}] - {attrs}"
    print(final,end=("\r","\n\n")[current==total])

real_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=(0,0),translate=(0,0),scale=(1,1),shear=(0,0,0,0),interpolation=transforms.InterpolationMode.BILINEAR),
])

fake_transform = transforms.Compose([
    transforms.ToTensor(),
])

PI = torch.tensor(3.14159265358979323)
batch_size = 32
num_epochs = 50
lrg = 0.01
lrd = 1e-4

dataset = datasets.MNIST(root="MNIST_data/", transform=real_transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

fake_dataset = datasets.MNIST(root="MNIST_data/", transform=fake_transform, download=True)
fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)

fixed_noise = torch.randint(0,len(loader)-1,())


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rot = torch.full((len(fake_dataset)//batch_size,batch_size,1),PI).float().requires_grad_()
        self.trans = torch.full((len(fake_dataset)//batch_size,batch_size,2),0.).float().requires_grad_()
        self.scale = torch.full((len(fake_dataset)//batch_size,batch_size,1),1.).float().requires_grad_()
        self.shear = torch.full((len(fake_dataset)//batch_size,batch_size,2),0.).float().requires_grad_()
    
    def forward(self,idx):
        for i, (batch, _) in enumerate(fake_loader,idx):
            full = None
            for j, img in enumerate(batch):
                img = img.unsqueeze(dim=0).float()
                img = affine(img,
                             self.rot[i][j],
                             self.trans[i][j],
                             self.scale[i][j],
                             self.shear[i][j])
                
                if full is None: full = img
                else: full = torch.cat((full,img))
            
            # torchviz.make_dot(full).render("zimgtest2",format="png"
            
            return full


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(28*28,1),
            nn.Sigmoid()
        )
    
    def forward(self,img):
        img = img.view(batch_size, -1)
        return self.net(img)

gen = Generator()
disc = Discriminator()
criterion = nn.BCELoss()

opt_gen = optim.Adam([gen.rot,gen.trans,gen.scale,gen.shear], lr=lrg)
opt_disc = optim.Adam(disc.parameters(), lr=lrd)


plot_image_grid(next(iter(loader))[0].reshape(-1, 28, 28, 1).numpy())
fake = gen(fixed_noise).reshape(-1, 28, 28, 1)
plot_image_grid(fake.detach().numpy())

autograd.set_detect_anomaly(True)   
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784)
        batch_size = real.shape[0]
        
        # Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        opt_disc.zero_grad()
        noise = torch.randint(0,len(loader)-1,())
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        # print("1",gen.rot.grad)
        opt_disc.step()
        
        # Train Generator min log(1-D(G(z))) <--> max log(D(G(z)))
        opt_gen.zero_grad()
        
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))  
        # torchviz.make_dot(lossG).render("zimgtest3",format="png")
        lossG.backward()
        print(gen.rot.grad[gen.rot.grad.nonzero(as_tuple=True)])
        """print(gen.trans.grad[gen.trans.grad.nonzero(as_tuple=True)])
        print(gen.scale.grad[gen.scale.grad.nonzero(as_tuple=True)])
        print(gen.shear.grad[gen.shear.grad.nonzero(as_tuple=True)])"""
        opt_gen.step()
        # with torch.no_grad(): gen.rot -= lrg * gen.rot.data
        with torch.no_grad():
            progress(batch_idx+1,len(loader),
                     g=round(float(lossG.mean().numpy()),2),
                     d=round(float(lossD.mean().numpy()),2),
                     r=round(float(gen.rot.median().numpy()),2),
                     t=round(float(gen.trans.median().numpy()),2),
                     sc=round(float(gen.scale.median().numpy()),2),
                     sh=round(float(gen.shear.median().numpy()),2))

    if epoch % 1 == 0:
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 28, 28, 1)
            plot_image_grid(fake.detach().numpy())
