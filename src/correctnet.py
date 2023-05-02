import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

import glob
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

def plot_image_grid(images, ncols=None, cmap="gray"):
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    _, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()): 
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)
    plt.show()

def progress(current,total,**kwargs):
    done_token, current_token = ("=", ">")
    token_arr = []
    bar = total // 1
    token_arr.extend([done_token]*(current//bar))
    if (total-current): token_arr.extend([current_token])
    attrs = json.dumps(kwargs).replace('"',"")[1:-1]
    final = f"{current}/{total} [{''.join(token_arr)}{' '*max(0,25-current//bar)}] - {attrs}"
    print(final,end=("\r","\n\n")[current==total])

root = "Affine/SkyFinder Affine/cameras/10870/"
PI = torch.tensor(3.14159265358979323)
CROP_SIZE = (128, 128)
batch_size = 2
num_batches = 4
num_epochs = 300
lrr = 0.1
lrt = 0.01
lrsc = 0.001
lrd = 0.00001

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(CROP_SIZE),
    transforms.ToTensor(),
])

dataset = []
for path in glob.glob(f"{root}/*.jpg"):
    try: dataset.append(transform(Image.open(path)))
    except OSError: pass
mean_pixel_values = [torch.mean(image) for image in dataset]

real_dataset = []
fake_dataset = []
for i in range(len(mean_pixel_values)):
    if mean_pixel_values[i] >= 0.5:
        real_dataset.append(dataset[i])
        fake_dataset.append(dataset[i])

for i in range(len(real_dataset)):
    img = real_dataset[i].unsqueeze(dim=0)
    r = torch.tensor(0.)
    t = torch.tensor([0., 0.])
    sc = torch.tensor(1.)
    rot = torch.stack([torch.stack([
        torch.stack([torch.cos(r) / sc, -torch.sin(r) / sc, torch.tensor(t[0]) / sc]),
        torch.stack([torch.sin(r) / sc, torch.cos(r) / sc, torch.tensor(t[1]) / sc])
        ])])
    grid = F.affine_grid(rot, img.size(), align_corners=False)
    img = F.grid_sample(img, grid, align_corners=False)
    real_dataset[i] = (img, 0)  

real_dataset = real_dataset[:-num_batches*(batch_size)]
print(len(real_dataset))
loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

fake_dataset = [(im, 0) for im in fake_dataset]
fake_dataset = fake_dataset[-num_batches*(batch_size):]
print(len(fake_dataset))
fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)

fixed_noise = torch.randint(0,num_batches-1,())


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rot = torch.full((len(fake_loader), batch_size),(2.)).requires_grad_()
        self.trans = torch.full((len(fake_loader),batch_size,2),(0.)).requires_grad_()
        self.scale = torch.full((len(fake_loader), batch_size),(1.5)).requires_grad_()
    
    def forward(self,idx):
        for i, (batch, _) in enumerate(fake_loader,idx):
            rot = torch.stack([torch.stack([
                torch.stack([torch.cos(self.rot[i][j]) / (self.scale[i][j]+1e-18), -torch.sin(self.rot[i][j]) / (self.scale[i][j]+1e-18), self.trans[i][j][0] / (self.scale[i][j]+1e-18)]),
                torch.stack([torch.sin(self.rot[i][j]) / (self.scale[i][j]+1e-18), torch.cos(self.rot[i][j]) / (self.scale[i][j]+1e-18), self.trans[i][j][1] / (self.scale[i][j]+1e-18)])
                ]) for j in range(batch_size)])
            grid = F.affine_grid(rot, batch.size(), align_corners=False)
            batch = F.grid_sample(batch, grid, align_corners=False)

            return batch


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3),
            nn.Conv2d(4,1,kernel_size=2),
            nn.AvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(3844,1),
            nn.Sigmoid()
        )
    
    def forward(self,img):
        return self.net(img)

gen = Generator()
disc = Discriminator()
criterion = nn.BCELoss()

opt_rot = optim.AdamW([gen.rot], lr=0.03)
opt_trans = optim.AdamW([gen.trans], lr=0.03)
opt_scale = optim.AdamW([gen.scale], lr=0.002)
opt_disc = optim.AdamW(disc.parameters(), lr=0.000001)

with torch.no_grad():
    plot_image_grid(next(iter(loader))[0].reshape(-1, 128, 128, 1).numpy())
    fake = gen(fixed_noise).reshape(-1, 128, 128, 1)
    plot_image_grid(fake.detach().numpy())


rot_array = []
trans_array = []
scale_array = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 1, 128, 128)
        
        # Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        opt_disc.zero_grad()
        noise = torch.randint(0,num_batches-1,())
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = lossD_real + lossD_fake
        lossD.backward(retain_graph=True)
        opt_disc.step()
        
        # Train Generator min log(1-D(G(z))) <--> max log(D(G(z)))
        opt_rot.zero_grad()
        opt_trans.zero_grad()
        opt_scale.zero_grad()
        
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))  
        lossG.backward()
        
        opt_rot.step()
        opt_trans.step()
        opt_scale.step()
        
        with torch.no_grad():
            progress(batch_idx+1,len(loader),
                     g=round(float(lossG.mean().numpy()),2),
                     d=round(float(lossD.mean().numpy()),2),
                     r=round(float(gen.rot.median().numpy()),2),
                     t=round(float(gen.trans.median().numpy()),2),
                     sc=round(float(gen.scale.median().numpy()),2))
            
    with torch.no_grad():
        rot_array.append(round(float(gen.rot.median().numpy()),2))
        trans_array.append(round(float(gen.trans.median().numpy()),2))
        scale_array.append(round(float(gen.scale.median().numpy()),2))

    if epoch % 25 == 0:
        with torch.no_grad():
            for i in range(len(fake_loader)):
                fake = gen(i).reshape(-1, 128, 128, 1)
                plot_image_grid(fake.detach().numpy())
