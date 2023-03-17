import torch 
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import warnings
from draw_rect import display_images_with_bounding_boxes, non_max_surpression
import os
import random

torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")





class TinyYOLOv2(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        anchors,
        network_type

    ):
        super().__init__()

        # Parameters
        self.network_type = network_type
        if network_type == "maxpool" or network_type == "maxpool_skip":
            self.forward = self.forward_maxpool
        elif network_type == "conv" or network_type == "conv_skip":
            self.forward = self.forward_conv
        else:
            raise Exception('Only possible network types are "maxpool", "maxpool_skip", "conv" and "conv_skip"')


        self.num_anchors = len(anchors)
        self.anchors = anchors
        #self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        # Layers
        if 'maxpool' in network_type:
            self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.slowpool = torch.nn.MaxPool2d(2, 1)
            self.pad = torch.nn.ReflectionPad2d((0, 1, 0, 1))
            self.norm1 = torch.nn.BatchNorm2d(16, momentum=0.1)
            self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1, bias=False)
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

            if 'skip' in network_type:
                self.conv8 = torch.nn.Conv2d(2048, 1024, 3, 1, 1, bias=False)
            else:
                self.conv8 = torch.nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)

            
            self.conv9 = torch.nn.Conv2d(1024, self.num_anchors * (5 + num_classes), 1, 1, 0)


        elif 'conv' in network_type:
            # My layers
            self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
            self.my_conv1 = torch.nn.Conv2d(3, 8, 3, 1, (3,1), bias=False)
            self.my_bn1 = torch.nn.BatchNorm2d(8, momentum=0.1)
            self.my_conv2 = torch.nn.Conv2d(8, 16, 3, 2, (3,1), bias=False)  #6,1
            self.my_bn2 = torch.nn.BatchNorm2d(16, momentum=0.1)
            self.my_conv3 = torch.nn.Conv2d(16, 32, 5, 2, (3,2), bias=False)
            self.my_bn3 = torch.nn.BatchNorm2d(32, momentum=0.1)

            self.my_conv4 = torch.nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            self.my_bn4 = torch.nn.BatchNorm2d(64, momentum=0.1)
            self.my_conv5 = torch.nn.Conv2d(64, 128, 3, 2, 1, bias=False)
            self.my_bn5 = torch.nn.BatchNorm2d(128, momentum=0.1)
            self.my_conv6 = torch.nn.Conv2d(128, 256, 5, 2, 2, bias=False)
            self.my_bn6 = torch.nn.BatchNorm2d(256, momentum=0.1)

            self.my_conv7 = torch.nn.Conv2d(256, 512, 3, 1, 1, bias=False)
            self.my_bn7 = torch.nn.BatchNorm2d(512, momentum=0.1)

            if 'skip' in network_type:
                self.my_conv8 = torch.nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
                self.my_bn8 = torch.nn.BatchNorm2d(1024, momentum=0.1)
            else:
                self.my_conv8 = torch.nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
                self.my_bn8 = torch.nn.BatchNorm2d(1024, momentum=0.1)

            self.my_conv9 = torch.nn.Conv2d(1024, 1024, 5, 2, 2, bias=False)
            self.my_bn9 = torch.nn.BatchNorm2d(1024, momentum=0.1)

            self.my_conv10 = torch.nn.Conv2d(1024, (self.num_anchors * (5 + num_classes)), 1, 1, 0, bias=False)


    def forward_maxpool(self, x):

        x = self.relu(self.pool(self.norm1(self.conv1(x))))
        
        x = self.relu(self.pool(self.norm2(self.conv2(x))))
        x = self.relu(self.pool(self.norm3(self.conv3(x))))
        
        if 'skip' in self.network_type:
            x_for_concat = x

        x = self.relu(self.pool(self.norm4(self.conv4(x))))
        x = self.relu(self.pool(self.norm5(self.conv5(x))))
        x = self.relu(self.slowpool(self.pad(self.norm6(self.conv6(x)))))
        x = self.relu(self.norm7(self.conv7(x)))

        if 'skip' in self.network_type:
            x_for_concat = x_for_concat.reshape((8, 1024, 12, 18))
            x = torch.cat([x, x_for_concat], dim=1)

        x = self.relu(self.norm8(self.conv8(x)))
        x = self.conv9(x)
        x = self.yolo(x)

        return x


    def forward_conv(self, x):

        x = self.relu(self.my_bn1(self.my_conv1(x)))
        x = self.relu(self.my_bn2(self.my_conv2(x)))
        x = self.relu(self.my_bn3(self.my_conv3(x)))

        if "skip" in self.network_type:
            x_for_concat = x

        x = self.relu(self.my_bn4(self.my_conv4(x)))
        x = self.relu(self.my_bn5(self.my_conv5(x)))
        x = self.relu(self.my_bn6(self.my_conv6(x)))
        x = self.relu(self.my_bn7(self.my_conv7(x)))

        if 'skip' in self.network_type:
            x_for_concat = x_for_concat.reshape((8, 512, 25, 36))
            x = torch.cat([x, x_for_concat], dim=1)

        x = self.relu(self.my_bn8(self.my_conv8(x)))
        x = self.relu(self.my_bn9(self.my_conv9(x)))
        x = self.my_conv10(x)
        
        x = self.yolo(x)

  
        return x


    def yolo(self, x):

        x = torch.permute(x, (0, 3, 2, 1)) # (Batch, width, height, len(anchors) * [5+num_classes])
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2], self.num_anchors, x.shape[3] // self.num_anchors))

        num_cells_width = 18
        num_cells_height = 12
        # [5 + num_classes = center_x, center_y, width, height, conf + ONE_HOT_CLASS_VECTOR]

        # Sigmoid gives us offset from cell's starting coordinates to bounding box center: for example 0.3
        # torch.arrange gives us the coordinates of cells start: for example 5.0
        # Combined they give us coordinate of bounding box in units of cells: for example 5.0 + 0.3 = 5.3
        # To plot this coordinate on original image you need to multiply it by cell cize
        cell_start_position_x = torch.arange(num_cells_width)[:, None, None]
        cell_start_position_y = torch.arange(num_cells_height)[:, None]

        cell_start_position_x = cell_start_position_x.to(x.device)
        cell_start_position_y = cell_start_position_y.to(x.device)

        x[..., 0] = x[..., 0].sigmoid() + cell_start_position_x
        x[..., 1] = x[..., 1].sigmoid() + cell_start_position_y
        x[..., 2] = x[..., 2].exp()
        x[..., 3] = x[..., 3].exp()
        x[..., 4] = x[..., 4].sigmoid()
        x[..., 5:] = x[..., 5:].softmax(-1)

        return x
    

def main():

    torch.manual_seed(1302)
    np.random.seed(1302)
    random.seed(1302)

    num_cells_height = 12 
    num_cells_width = 18
    cell_height = 32
    cell_width = 32
    anchor_height = 250
    anchor_width = 300
    

    torch.manual_seed(1302)
    tyv2 = TinyYOLOv2()
    rook = torch.Tensor(np.array(Image.open("/media/workstation/Disk 1/cropped_images_small/14_udar_munje_004_high.jpg")))
    rook = torch.permute(rook, (2, 0, 1))
    rook = rook[None, :, :, :]
    print(f'image shape = {rook.shape}')
    output = tyv2.forward(rook)
    print(f'output shape = {output.shape}')
    non_max_surpression(output)
    exit(-1)
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

    jeremija_label = labels['jeremija']
    loss = tyv2.loss(output, jeremija_label)
    print(loss)
    exit()

    images = load_images(images_dir)
#
    for image_key in images:
        image = images[image_key]
        label = labels[image_key]
        
        #label = label.view(label.shape[0] * label.shape[1] * label.shape[2], label.shape[3])
        label = label.view(-1, label.shape[3])
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

    
if __name__ == '__main__':
    main()