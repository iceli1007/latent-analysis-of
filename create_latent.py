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
kkk=9
#from torch.nn import init
from torch.nn import init
from torch.nn import utils
kkk=9
os.makedirs("images", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
data_path='/data4/lzq/data/2019/fashion-mnist/data/imgs/'
data_file_path='/data4/lzq/data/2019/fashion-mnist/data/0.txt'
model_path='/data4/lzq/tf/GAN/DCGAN/model2.pth'
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
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
print(cuda)
tt=0
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
]) 
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

class Embeding(nn.Module):
    def __init__(self):
        super(Embeding, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4+1 
        #print(ds_size)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 100))
     
    def forward(self, img):
        #print(np.shape(img))
        out = self.model(img)
        #print(np.shape(out))
        out = out.view(out.shape[0], -1)
        #print(np.shape(out))
        validity = self.adv_layer(out)
        
        return validity



generator = Generator()
discriminator = Embeding()
#discriminator = Embeding(num_features=64,num_classes=0,activation=F.relu)
if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights

loader=torch.load(model_path)
state=loader['G']
generator.load_state_dict(state,strict=False)
#print(generator)
generator.eval()

Eloader=torch.load('/data4/lzq/tf/GAN/DCGAN/cgan_with_projection/cgan_embeding/218400.pth')
state=Eloader['D']
discriminator.load_state_dict(state)
discriminator.eval()
for j in range(10):

    with open('/data4/lzq/data/2019/fashion-mnist/data/%d.txt' % j, 'rb') as f:
        data = f.readlines()
    datalist_0 = [k[0] for k in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), data)]

    img_list=[]
    latents=[]
    for i in datalist_0:
        img=io.imread(data_path+i,as_grey=True)
        img_list.append(img)
    for img, id in zip(img_list,datalist_0):
        #mpimg.imsave('/home/lzq/tf/GAN/DCGAN/%da.jpg' % tt,img)
        img=preprocess(img)
        #print(img)
        img=img.to(device).float()
        img=torch.unsqueeze(img,0)
        latent_space=discriminator(img)
        h=latent_space.cpu().detach().numpy()
        h=h.reshape(128)
        print(id)
        latents.append(h)
    latents=np.array(latents)
    print(np.shape(latents))
    ##save the latent vector of real images.
    np.savetxt('/data4/lzq/tf/GAN/DCGAN/cgan_with_projection/cgan_embeding/latent_%d' %j,latents)
    #a=np.loadtxt('latent_%d' % j)
    
           
