import argparse
import os
import numpy as np
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
	
from skimage import io
from PIL import Image
from skimage import io
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import random
kk=9

os.makedirs("images", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser()
data_path='/data4/lzq/data/2019/fashion-mnist/data/imgs/'
data_file_path='/home/lzq/tf/GAN/DCGAN/latent_0'
GAN_model_path='/home/lzq/tf/GAN/DCGAN/211600.pth'
classfication_mode_path='/home/lzq/tf/GAN/DCGAN/fashion-mnist/saved-models/FashionSimpleNet-run-0.pth.tar'

parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--path", type=str, default='/data4/lzq/data/2019/fashion-mnist/data/imgs/')
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

latent_date=np.loadtxt(data_file_path)
print(np.shape(latent_date))
np.random.shuffle(latent_date)
print(np.shape(latent_date))
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4  #7
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
class FashionSimpleNet(nn.Module):

    """ Simple network"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        #print(np.shape(x))
        x = self.features(x)
        #print(np.shape(x))
        x = x.view(x.size(0), 64 * 7 * 7)
        #print(np.shape(x))
        x = self.classifier(x)
        #print(np.shape(x))
        return x
G_model=Generator()
cls_model=FashionSimpleNet()
loader=torch.load(GAN_model_path)
state=loader['G']
G_model.load_state_dict(state,strict=False)
G_model.eval()
loader_c=torch.load(classfication_mode_path)
state=loader_c['state_dict']
alpha=0.1
lamda_1=10
lamda_2=0.2
cls_model.load_state_dict(state)
cls_model.eval()



filepath='0_3/'
ttt=3
#weight=np.random.rand(1,100)
weight=0.5*np.ones([1,100])
weight=torch.from_numpy(weight)
weight=Variable(weight,requires_grad=True).float()
#mean_latent=np.mean(latent_date,axis=0)
#print(mean_latent)
#mean_latent=mean_latent.reshape(1,100)
#constant_K=1*np.ones([1,100])
#constant_K=constant_K*mean_latent
#constant_K=torch.from_numpy(constant_K).float()


ones=torch.ones(1,100).float()
#G_model.cuda()
#cls_model.cuda()
epoch=1


#print(weight)
for j in range(epoch):
    for i in range(7000):
        input=latent_date[i][:]
        #print(input)
        mean_latent=np.mean(input)
        input=np.reshape(input,(1,100))
        input=torch.from_numpy(input).float()

    
        


        weight_1=torch.tanh(weight)
        print(weight_1)
        deltazheng=random.randint(20,40)
        deltafu=random.randint(20,40)
        input_1=torch.div(torch.mul(2*ones-weight_1*mean_latent*20,input),2)
        input_2=torch.div(torch.mul(2*ones+weight_1*mean_latent*20,input),2)
        mid=G_model(input)
        mid_1=G_model(input_1)
        output_1=cls_model(mid_1)
        mid_2=G_model(input_2)
        output_2=cls_model(mid_2)
        output=cls_model(mid)
        
        output_1=torch.nn.Softmax()(output_1)[0,ttt]
        output_2=torch.nn.Softmax()(output_2)[0,ttt]
        output=torch.nn.Softmax()(output)[0,ttt]
        
        print(output)
        print(output_1)
        print(output_2)
        #loss=torch.div(-torch.abs(output_1-output)-torch.abs(output_2-output),2)+lamda_2* torch.norm(weight_1,2)
        loss=-torch.max(torch.abs(output_1-output),torch.abs(output_2-output))+lamda_2* torch.norm(weight_1,2)
        grad=torch.autograd.grad(loss,weight)

        weight=weight-0.05*grad[0]
    
weight_1=torch.tanh(weight)
weight_1=weight_1.detach().numpy()
weight_1=np.reshape(weight_1,(100))
import os
if not os.path.exists('/home/lzq/tf/GAN/DCGAN/result_end/'+filepath):
    os.makedirs('/home/lzq/tf/GAN/DCGAN/result_end/'+filepath)
np.savetxt('/home/lzq/tf/GAN/DCGAN/result_end/'+filepath+'weight_4',weight_1)

for i in weight_1:
    print(i)



