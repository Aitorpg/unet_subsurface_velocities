# Adapted from: https://github.com/milesial/Pytorch-UNet
#Changes involve 
# 1) adding an Nbase parameter which changes the UNet
# size without changing the topology and
# 2) Adding an input Batch normalization layer
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, Nbase = 16):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inpBN  = nn.BatchNorm2d(n_channels) #****May need to delete and normalize the data before training
        self.inc = (DoubleConv(n_channels, Nbase))
        self.down1 = (Down(Nbase, Nbase*2))
        self.down2 = (Down(Nbase*2, Nbase*4))
        self.down3 = (Down(Nbase*4, Nbase*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(Nbase*8, Nbase*16 // factor))
        self.up1 = (Up(Nbase*16, Nbase*8 // factor, bilinear))
        self.up2 = (Up(Nbase*8, Nbase*4 // factor, bilinear))
        self.up3 = (Up(Nbase*4, Nbase*2 // factor, bilinear))
        self.up4 = (Up(Nbase*2, Nbase, bilinear))
        self.outc = (OutConv(Nbase, n_classes))

    def forward(self, x):
        x1 = self.inc(self.inpBN(x))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_nobatchnorm(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, Nbase = 16):
        super(UNet_nobatchnorm, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #self.inpBN  = nn.BatchNorm2d(n_channels) 
        self.inc = (DoubleConv(n_channels, Nbase))
        self.down1 = (Down(Nbase, Nbase*2))
        self.down2 = (Down(Nbase*2, Nbase*4))
        self.down3 = (Down(Nbase*4, Nbase*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(Nbase*8, Nbase*16 // factor))
        self.up1 = (Up(Nbase*16, Nbase*8 // factor, bilinear))
        self.up2 = (Up(Nbase*8, Nbase*4 // factor, bilinear))
        self.up3 = (Up(Nbase*4, Nbase*2 // factor, bilinear))
        self.up4 = (Up(Nbase*2, Nbase, bilinear))
        self.outc = (OutConv(Nbase, n_classes))

    def forward(self, x):
        #x1 = self.inc(self.inpBN(x))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
