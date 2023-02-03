from PIL import ImageDraw
import torchvision
from IPython.display import display
import torch
from matplotlib import pyplot as plt
import matplotlib
from collections import defaultdict
import os
from math import floor

import torch 
import numpy as np
from PIL import Image


def non_max_surpression(predictions, confidence_treshold=0.5):
    '''
    Surpress any predictions with confidence less than confidence_treshold
    In this task there is a maximum of one detection per class so we only take the prediction with maximum confidence per class

    Parameters:
        - predictions (torch.Tensor): prediction vector which will be surpressed
            this vector has a shape of (B, W, H, (5 + Num_Classes))
        - confidence_treshold (float: between 0 and 1): minimum confidence required for prediction
            all predictions with lower confidence than this will be surpressed
    '''
    device = predictions.device
    # Surpressing all predictions with confidence less than confidence_treshold
    zeros_vector = torch.zeros((predictions.shape[-1])).to(device)
    confidences = predictions[..., 4]
    predictions[confidences < confidence_treshold] = zeros_vector

    class_probabilities = predictions[..., 5:]
    
    maxes, max_indices = class_probabilities.max(dim=-1)
    maxes = maxes[..., None]
    class_one_hot = (class_probabilities == maxes)
    
    num_classes = class_probabilities.shape[-1]
    for class_index in range(num_classes):
        same_class = class_one_hot[..., class_index]
        conf_times_class = confidences * same_class
        class_conf_maxes = conf_times_class.max()
        predictions[torch.logical_and(conf_times_class > 0, conf_times_class != class_conf_maxes)] = zeros_vector

    

def display_images_with_bounding_boxes(image, bounding_boxes, cell_width, cell_height, anchor_width, anchor_height, name=''):
    '''
    Displays images with bounding boxes around them

    Input:
        - image (Tensor of shape: ): image for display
        - bounding_boxes (Tensor containing bounding box tensors): bounding boxes which will be displayed
            each bounding box is tensor of shape: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
            CLASS_ONE_HOT_ENCODING contains one element for each class 
    '''

    image = torch.permute(image, (1, 2, 0)).int()
    fig, axis = plt.subplots()
    image = image.detach().cpu().numpy()
    axis.imshow(image)
    bounding_boxes = bounding_boxes.detach().cpu().numpy()

    bounding_boxes = bounding_boxes.reshape(-1, bounding_boxes.shape[-1])
    
    
    for bounding_box in bounding_boxes:
        

        if not bounding_box.any():
            continue
        
        # Bounding box contains: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
        # Box_center_x, and Box_center_y are in units of cells so we need to multiply them to display them on image
        # box_width and box_height are in units of anchor sizes so they too need to be multiplied in order to be displayed

        box_center_x = bounding_box[0] * cell_width
        box_center_y = bounding_box[1] * cell_height 

        box_width  = bounding_box[2] * anchor_width 
        box_height = bounding_box[3] * anchor_height

        # Class Rectangle needs top left corner as input
        top_left_corner_x = box_center_x - box_width  / 2
        top_left_corner_y = box_center_y - box_height / 2

        edge_color = 'red'

        rect = matplotlib.patches.Rectangle((top_left_corner_x, top_left_corner_y), box_width, box_height, fill=False, edgecolor=edge_color) 
        axis.add_patch(rect)

    plt.show()
        