import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import ssl
import torch

ssl._create_default_https_context = ssl._create_unverified_context
__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

      

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, dilation = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation = 1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, dilation = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, dilation = 1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1, dilation = 1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1, dilation = 1)

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1, dilation = 1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1, dilation = 1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1, dilation = 1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1, dilation = 1)
        
        self.conv13 = nn.Conv2d(512, 512, 3, padding=2, dilation = 2)
        self.conv14 = nn.Conv2d(512, 512, 3, padding=2, dilation = 2)
        self.conv15 = nn.Conv2d(512, 512, 3, padding=2, dilation = 2)
        self.conv16 = nn.Conv2d(512, 512, 3, padding=2, dilation = 2)

        initial_weights = model_zoo.load_url(model_urls['vgg19'])        
        self.conv1.weight.data = initial_weights['features.0.weight']        
        self.conv1.bias.data =  initial_weights['features.0.bias'] 
        self.conv2.weight.data = initial_weights['features.2.weight']        
        self.conv2.bias.data =  initial_weights['features.2.bias']       
        self.conv3.weight.data = initial_weights['features.5.weight']        
        self.conv3.bias.data =  initial_weights['features.5.bias']
        self.conv4.weight.data = initial_weights['features.7.weight']        
        self.conv4.bias.data =  initial_weights['features.7.bias']
        self.conv5.weight.data = initial_weights['features.10.weight']        
        self.conv5.bias.data =  initial_weights['features.10.bias']
        self.conv6.weight.data = initial_weights['features.12.weight']        
        self.conv6.bias.data =  initial_weights['features.12.bias']
        self.conv7.weight.data = initial_weights['features.14.weight']        
        self.conv7.bias.data =  initial_weights['features.14.bias']
        self.conv8.weight.data = initial_weights['features.16.weight']        
        self.conv8.bias.data =  initial_weights['features.16.bias']
        self.conv9.weight.data = initial_weights['features.19.weight']        
        self.conv9.bias.data =  initial_weights['features.19.bias']
        self.conv10.weight.data = initial_weights['features.21.weight']        
        self.conv10.bias.data =  initial_weights['features.21.bias']
        self.conv11.weight.data = initial_weights['features.23.weight']        
        self.conv11.bias.data =  initial_weights['features.23.bias']
        self.conv12.weight.data = initial_weights['features.25.weight']        
        self.conv12.bias.data =  initial_weights['features.25.bias']
        self.conv13.weight.data = initial_weights['features.28.weight']        
        self.conv13.bias.data =  initial_weights['features.28.bias']
        self.conv14.weight.data = initial_weights['features.30.weight']        
        self.conv14.bias.data =  initial_weights['features.30.bias']
        self.conv15.weight.data = initial_weights['features.32.weight']        
        self.conv15.bias.data =  initial_weights['features.32.bias']
        self.conv16.weight.data = initial_weights['features.34.weight']        
        self.conv16.bias.data =  initial_weights['features.34.bias'] 
       
        self.stage1 = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)    
        )
        
        
        self.stage2 = nn.Sequential(
            self.conv3,
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)       
        )
            
        self.stage3 = nn.Sequential(
            self.conv5,
            nn.ReLU(inplace=True),
            self.conv6,
            nn.ReLU(inplace=True),
            self.conv7,
            nn.ReLU(inplace=True),
            self.conv8,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)    
        )
        
        self.stage4 = nn.Sequential(
            self.conv9,
            nn.ReLU(inplace=True),
            self.conv10,
            nn.ReLU(inplace=True),
            self.conv11,
            nn.ReLU(inplace=True),
            self.conv12,
            nn.ReLU(inplace=True),
        )

        self.stage5 = nn.Sequential(
            self.conv13,
            nn.ReLU(inplace=True),
            self.conv14,
            nn.ReLU(inplace=True),
            self.conv15,
            nn.ReLU(inplace=True),
            self.conv16,
            nn.ReLU(inplace=True),
        )     

        self.reg_branch = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1), 
            nn.ReLU()
        )

        self.seg_branch = nn.Sequential(
            nn.Conv2d(512, 1, 1), 
            nn.ReLU(),
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       
        self.sigmoid = nn.Sigmoid()

    def forward(self, x ):

        x_min = self.pool(x)
        x_max = F.upsample_bilinear(x, scale_factor=2)

        # x traing
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x_map = self.reg_branch(x)
        x_mask = self.seg_branch(x)      

   
       
        return  x_map, x_mask


def vgg19():
    model = VGG()
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    total = sum([x.nelement() for x in model.parameters()])
    print('params ==', total / 1e6)
    return model

