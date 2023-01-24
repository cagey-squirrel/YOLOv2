import torch 
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import warnings
from draw_rect import show_images_with_boxes, load_annotations, get_labels_from_annotations, load_images, display_images_with_bounding_boxes
from loss_functions import YOLOLoss
import os

warnings.filterwarnings("ignore")

class TinyYOLOv2(torch.nn.Module):
    def __init__(
        self,
        num_classes=6,
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
        self.conv1 = torch.nn.Conv2d(4, 16, 3, 1, 1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.norm4 = torch.nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.norm5 = torch.nn.BatchNorm2d(256, momentum=0.1)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.norm6 = torch.nn.BatchNorm2d(512, momentum=0.1)
        self.conv6 = torch.nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.norm7 = torch.nn.BatchNorm2d(1024, momentum=0.1)
        self.conv7 = torch.nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.norm8 = torch.nn.BatchNorm2d(1024, momentum=0.1)
        self.conv8 = torch.nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
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
        #if yolo:
        #    x = self.yolo(x)
        # shape of x is now (Batch_size, [5+num_classes], height_in_cells, width_in_cells)
        # 1 11 12 18
        #print(x.shape)
        x = self.my_yolo(x)
        
        return x

    def my_yolo(self, x, for_output=True):
        x = torch.permute(x, (0, 3, 2, 1)) # (Batch, width, height, [5+num_classes])

        num_cells_width = 18
        num_cells_height = 12
        # [5 + num_classes = center_x, center_y, width, height, conf + ONE_HOT_CLASS_VECTOR]
        x[..., 0] = x[..., 0].sigmoid()
        x[..., 1] = x[..., 1].sigmoid()
        x[..., 2] = x[..., 2].exp()
        x[..., 3] = x[..., 3].exp()
        x[..., 4] = x[..., 4].sigmoid()
        x[..., 5:] = x[..., 5:].softmax(-1)

        #print(f'basic x shape = {x.shape}')
        #print(f'x shape = {x[..., 0].shape}')
        #print(f'torch shape = {torch.arange(cells_width).shape}')
        #exit(-1)
        if for_output:
            x[..., 0] += torch.arange(num_cells_width)[None, :, None]
            x[..., 1] += torch.arange(num_cells_height)[None, None, :]

        #print(torch.arange(num_cells_width)[None, :, None])
        #print(torch.arange(num_cells_height)[None, None, :])
        #print(x[...,0:2])
        return x
    
    def loss(self, x, labels):

        # Loss = loss_class + loss_conf + loss_coord
        # Od svih predikcija u jednoj celiji treba izabrati najsigurniju -> ovo je odradjeno jer ima samo jedna predikcija po celiji
        # Od svih anchora u jednoj celiji treba izabrati onaj koji se najbolje uklapa u labelu -> ovo je odradjeno jer ima samo jedan anchor po celiji

        # Dakle treba odrediti u kojoj celiji je objekat a u kojoj nije, ovo moze i kroz labele
        # Treba primeniti odgovarajuci loss na sve delove

        # Recimo da labels stize u formatu (B, A, W, H, (Cx, Cy, Wx, Wy, ONE_HOT_CLASS_ENCODINGS))
        # Shape is (B, A, W, H, 5+C)

        # Sta uzimamo za loss: Uzimamo rastojanje od gornjeg levog ugla do sredine objekta u celiji 
        # U labeli je meni dato za sreidnu slike vrednost centra npr 0.55, rastojanje od gornjeg 

        object_present = labels[..., 5] # getting the certainty from labels: 1 for object present, 0 for object absent 
        object_not_present = torch.logical_not(object_present)


    
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

        #print(range_x)
  
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
        #print(x[:, :, :, :, 0, None].squeeze())
        #print(range_x[None,None,:,:,None].squeeze())

        return x # (batch_size, # anchors, height, width, 5+num_classes)
    

def main():

    num_cells_height = 12 
    num_cells_width = 18
    cell_height = 32
    cell_width = 32
    anchor_height = 250
    anchor_width = 300
    


    torch.manual_seed(1302)
    tyv2 = TinyYOLOv2()
    rook = torch.Tensor(np.array(Image.open("all_same_size_imgs/jeremija.png")))
    rook = torch.permute(rook, (2, 0, 1))
    rook = rook[None, :, :, :]
    print(f'image shape = {rook.shape}')
    output = tyv2.forward(rook)
    #display_images_with_bounding_boxes(rook[0], output[0, 0, :, :], cell_width, cell_height, anchor_width, anchor_height)
    #exit(-1)
    #print(f'output shape is {output.shape}')
    #exit(-1)
    #show_images_with_boxes(rook, output)
    #print(f'example of output is: {output[0, 0, 0, 0, :]}')

    images_dir = 'all_same_size_imgs'
    annotations_path = 'annotations/annotations2.csv'
    
    
    
    annotations = load_annotations(annotations_path)
    labels = get_labels_from_annotations(annotations, 389, 588)

    images = load_images(images_dir)
#
    for image_key in images:
        image = images[image_key]
        label = labels[image_key]
        
        label = label.view(label.shape[0] * label.shape[1] * label.shape[2], label.shape[3])
        print(f'label_shape = {label.shape}')
        display_images_with_bounding_boxes(image, label, cell_width, cell_height, anchor_width, anchor_height, name=image_key)



    #print(labels)
    exit(-1)

    jeremija_label = labels['jeremija']
    jeremija_label = jeremija_label.squeeze()
    jeremija_label = jeremija_label.tolist()
    jeremija_labels = [jeremija_label for _ in range(12*18-1)]
    jeremija_labels = np.array(jeremija_labels)
    #print(jeremija_labels)
    jeremija_labels = torch.Tensor(jeremija_labels)
    zeross = torch.zeros((1,12))
    zeross[..., -1] = -1
    jeremija_labels = torch.cat([jeremija_labels, zeross], 0)
    jeremija_labels = jeremija_labels[None, ...]


    #raw_output = tyv2.forward(rook, yolo=False)
    #loss = YOLOLoss()
    #loss_val = loss.forward(raw_output, jeremija_labels)
    #print(loss_val)

    #output = tyv2.forward(rook, yolo=True)
    #show_images_with_boxes(rook, output)

    

main()