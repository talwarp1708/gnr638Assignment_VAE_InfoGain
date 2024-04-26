import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture proposed for InfoGAN for PatternNet
"""

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.tconv1 = nn.ConvTranspose2d(826, 1024, 2, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(1024)
    
    self.tconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(512)
    
    self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(256)
    
    self.tconv4 = nn.ConvTranspose2d(256, 256, 4, 2, padding=1, bias=False)
    self.bn4 = nn.BatchNorm2d(256)
    
    self.tconv5 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)
    self.bn5 = nn.BatchNorm2d(128)
    
    self.tconv6 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
    
    self.tconv7 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False)
    
    self.tconv8 = nn.ConvTranspose2d(32, 3, 4, 2, padding=1, bias=False)

  def forward(self, x):
    x = F.relu(self.bn1(self.tconv1(x)))
    #print("G: shape after tconv1:" ,x.shape)
    x = F.relu(self.bn2(self.tconv2(x)))
    #print("G: shape after tconv2:" ,x.shape)
    x = F.relu(self.bn3(self.tconv3(x)))
    #print("G: shape after tconv3:" ,x.shape)
    x = F.relu(self.bn4(self.tconv4(x)))
    #print("G: shape after tconv4:" ,x.shape)
    x = F.relu(self.bn5(self.tconv5(x)))
    #print("G: shape after tconv5:" ,x.shape)
    x = F.relu(self.tconv6(x))
    #print("G: shape after tconv6:" ,x.shape)
    x = F.relu(self.tconv7(x))
    #print("G: shape after tconv7:" ,x.shape)

    img = torch.tanh(self.tconv8(x))
      
    #print("G: shape after tconv8:" ,img.shape)
    return img

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

    self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(128)

    self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
    self.bn3 = nn.BatchNorm2d(256)

    self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
    self.bn4 = nn.BatchNorm2d(512)

    self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
    self.bn5 = nn.BatchNorm2d(1024)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
    #print("D: shape after conv1:" ,x.shape)
    x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
    #print("D: shape after conv2:" ,x.shape)
    x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
    #print("D: shape after conv3:" ,x.shape)
    x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
    #print("D: shape after conv4:" ,x.shape)
    x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
    #print("D: shape after conv5:" ,x.shape)
    return x

class DHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1024, 1, 4)
		self.conv2 = nn.Conv2d(1,1,5)

	def forward(self, x):
		#print("Input Shape for DHead: ", x.shape)
		output = torch.sigmoid(self.conv1(x))
		output = torch.sigmoid(self.conv2(output))
		#print("Output Shape for DHead: ", output.shape)
		return output

class QHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1024, 512, 4, bias=False)
		self.bn1 = nn.BatchNorm2d(512)
		self.conv2 = nn.Conv2d(512, 256, 4, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		#self.conv3 = nn.Conv2d(256, 128, 4, bias=False)
		#self.bn3 = nn.BatchNorm2d(128)

		self.conv_disc = nn.Conv2d(256, 570, 2)

		self.conv_mu = nn.Conv2d(256, 1, 1)
		self.conv_var = nn.Conv2d(256, 1, 1)

	def forward(self, x):
		#print("QHead Input shape: ", x.shape)
		x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
		#print("QHead shape after conv1: ", x.shape)
		x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
		#print("QHead shape after conv2: ", x.shape)
		disc_logits = self.conv_disc(x).squeeze()
		#print("QHead disc_logits shape: ", x.shape)
		# Not used during training for celeba dataset.
		mu = self.conv_mu(x).squeeze()
		#print("QHead mu shape: ", x.shape)
		var = torch.exp(self.conv_var(x).squeeze())
		#print("QHead var shape: ", x.shape)
		return disc_logits, mu, var
