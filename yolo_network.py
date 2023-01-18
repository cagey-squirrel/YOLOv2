import torch 
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import warnings
from draw_rect import show_images_with_boxes

warnings.filterwarnings("ignore")

class TinyYOLOv2(torch.nn.Module):
    def __init__(
        self,
        num_classes=4,
        anchors=(
            (1.08, 1.19),
            #(3.42, 4.41),
            #(6.63, 11.38),
            #(9.42, 5.11),
            #(16.62, 10.52),
        ),
    ):
        super().__init__()

        # Parameters
        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        # Layers
        self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.slowpool = torch.nn.MaxPool2d(2, 1)
        self.pad = torch.nn.ReflectionPad2d((0, 1, 0, 1))
        self.norm1 = torch.nn.BatchNorm2d(16, momentum=0.1)
        self.conv1 = torch.nn.Conv2d(3, 16, 5, 1, 1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, 1, 1, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = torch.nn.Conv2d(32, 64, 5, 1, 1, bias=False)
        self.norm4 = torch.nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = torch.nn.Conv2d(64, 128, 5, 1, 1, bias=False)
        self.norm5 = torch.nn.BatchNorm2d(256, momentum=0.1)
        self.conv5 = torch.nn.Conv2d(128, 256, 5, 1, 1, bias=False)
        self.norm6 = torch.nn.BatchNorm2d(512, momentum=0.1)
        self.conv6 = torch.nn.Conv2d(256, 512, 5, 1, 1, bias=False)
        self.norm7 = torch.nn.BatchNorm2d(1024, momentum=0.1)
        self.conv7 = torch.nn.Conv2d(512, 1024, 5, 1, 1, bias=False)
        self.norm8 = torch.nn.BatchNorm2d(1024, momentum=0.1)
        self.conv8 = torch.nn.Conv2d(1024, 1024, 5, 10, 1, bias=False)
        self.conv9 = torch.nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):
        x = self.relu(self.pool(self.norm1(self.conv1(x))))
        x = self.relu(self.pool(self.norm2(self.conv2(x))))
        x = self.relu(self.pool(self.norm3(self.conv3(x))))
        x = self.relu(self.pool(self.norm4(self.conv4(x))))
        x = self.relu(self.pool(self.norm5(self.conv5(x))))
        x = self.relu(self.slowpool(self.pad(self.norm6(self.conv6(x)))))
        x = self.relu(self.norm7(self.conv7(x)))
        x = self.relu(self.norm8(self.conv8(x)))
        x = self.conv9(x)
        if yolo:
            x = self.yolo(x)
        return x
    
    def yolo(self, x):

        # x is of size: grid, grid, anchors * (5 + num_classes)
        # broj slika, (5+num_classes), nh, nw


        # store the original shape of x
        nB, _, nH, nW = x.shape
        
        # reshape the x-tensor: (batch size, # anchors, height, width, 5+num_classes)
        x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

        # get normalized auxiliary tensors
        anchors = self.anchors.to(dtype=x.dtype, device=x.device)
        anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

        range_y, range_x = torch.meshgrid(
            torch.arange(nH, dtype=x.dtype, device=x.device),
            torch.arange(nW, dtype=x.dtype, device=x.device),
        )
  
        # compute boxes.
        x = torch.cat([
            (x[:, :, :, :, 0, None].sigmoid() + range_x[None,None,:,:,None]) / nW,  # X center
            (x[:, :, :, :, 1, None].sigmoid() + range_y[None,None,:,:,None]) / nH,  # Y center
            (x[:, :, :, :, 2, None].exp() * anchor_x[None,:,None,None,None]) / nW,  # Width
            (x[:, :, :, :, 3, None].exp() * anchor_y[None,:,None,None,None]) / nH,  # Height
            x[:, :, :, :, 4, None].sigmoid(), # confidence
            x[:, :, :, :, 5:].softmax(-1), # classes
        ], -1)

        # Za svaku celiju imamo po predikciju za centar x-a 
        # Ta predikcija pomocu relu ide izmedju 0 i 1 i govori koliko je pomerena od centra izrazeno u procentima od sirine 
        print(x[:, :, :, :, 0, None].squeeze())
        print(range_x[None,None,:,:,None].squeeze())

        return x # (batch_size, # anchors, height, width, 5+num_classes)

def main():
    torch.manual_seed(1302)
    tyv2 = TinyYOLOv2()
    rook = torch.Tensor(np.array(Image.open("rook.jpg")))
    rook = torch.permute(rook, (2, 0, 1))
    rook = rook[None, :, :, :]
    output = tyv2.forward(rook)
    print(f'output shape is {output.shape}')
    show_images_with_boxes(rook, output)
    #print(f'example of output is: {output[0, 0, 0, 0, :]}')

main()