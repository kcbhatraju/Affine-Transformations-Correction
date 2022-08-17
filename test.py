from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchviz
from torchvision import transforms, datasets
from diffaff import affine

img = Image.open("sample.png")
t = transforms.ToTensor()
img = t(img).unsqueeze(dim=0).double()
print(img.shape)
rot = torch.full((2,2,1),3.14).double().requires_grad_()
trans = torch.full((2,2,2),0.).double().requires_grad_()
scale = torch.full((2,2,1),1.).double().requires_grad_()
shear = torch.full((2,2,2),0.).double().requires_grad_()

full = affine(img,rot[0][0],trans[0][0],scale[0][0],shear[0][0])
full.mean().backward()

print(rot.grad)
print(trans.grad)
print(scale.grad)
print(shear.grad)

plt.imshow(transforms.ToPILImage()(full.squeeze(dim=0).detach()))
plt.show()
