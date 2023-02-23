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


def display_images_with_bounding_boxes(image, bounding_boxes, classes, cell_width, cell_height, anchor_width, anchor_height, name=''):
    '''
    Displays images with bounding boxes around them

    Input:
        - image (Tensor of shape: ): image for display
        - bounding_boxes (Tensor containing bounding box tensors): bounding boxes which will be displayed
            each bounding box is tensor of shape: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
            CLASS_ONE_HOT_ENCODING contains one element for each class 
    '''

    image = torch.permute(image, (1, 2, 0))
    fig, axis = plt.subplots()
    image = image.detach().cpu().numpy()
    axis.imshow(image)
    image_height = image.shape[0]
    image_width = image.shape[1]
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

        class_probabilities = bounding_box[5:]
        class_max_index = class_probabilities.argmax()
        object_class = classes[class_max_index]

        # Making sure the box fits the image (doesnt go beyond)
        top_left_corner_x = min(max(1, top_left_corner_x), image_width-1)
        top_left_corner_y = min(max(1, top_left_corner_y), image_height-1)

        rect = matplotlib.patches.Rectangle((top_left_corner_x, top_left_corner_y), box_width, box_height, fill=False, edgecolor=edge_color) 
        axis.text(x=top_left_corner_x + 10, y = top_left_corner_y + 20, s = object_class, color = 'red')
        axis.add_patch(rect)

    plt.show()


def add_bounding_boxes_to_axis(bounding_boxes, axis, classes, height_and_width_info, color):

    image_height, image_width, cell_width, cell_height, anchor_width, anchor_height = height_and_width_info
    bounding_boxes = bounding_boxes.reshape(-1, bounding_boxes.shape[-1])
    
    for bounding_box in bounding_boxes:

        if bounding_box[4] < 0.1:
            continue

        # Bounding box contains: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
        # Box_center_x, and Box_center_y are in units of cells so we need to multiply them to display them on image
        # box_width and box_height are in units of anchor sizes so they too need to be multiplied in order to be displayed
        #print(f'box_center_x = {bounding_box[0]} box_center_y = {bounding_box[1]}')
        box_center_x = bounding_box[0] * cell_width
        box_center_y = bounding_box[1] * cell_height 

        box_width  = bounding_box[2] * anchor_width 
        box_height = bounding_box[3] * anchor_height

        
        # Class Rectangle needs top left corner as input
        top_left_corner_x = box_center_x - box_width  / 2
        top_left_corner_y = box_center_y - box_height / 2

        class_probabilities = bounding_box[5:]
        class_max_index = class_probabilities.argmax()
        object_class = classes[class_max_index]

        # Making sure the box fits the image (doesnt go beyond)
        top_left_corner_x = min(max(1, top_left_corner_x), image_width-1)
        top_left_corner_y = min(max(1, top_left_corner_y), image_height-1)

        rect = matplotlib.patches.Rectangle((top_left_corner_x, top_left_corner_y), box_width, box_height, fill=False, edgecolor=color) 
        axis.text(x=top_left_corner_x + 10, y = top_left_corner_y + 20, s = object_class, color = 'red')
        axis.add_patch(rect)


def output_predictions(images, labels_list, predictions, images_names, epoch_num, params):
    '''
    Outputs prediction detections and true label boxes on images

    Input:
        - images:
        - labels
        - predictions
        - images_names
    '''

    height_and_width_info, output_dir_path, train_text_file, classes = params

    output_dir = os.path.join(output_dir_path, str(epoch_num))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    images = images.detach().cpu().numpy()
    labels_list = labels_list.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    for image, labels, prediction, image_name in zip(images, labels_list, predictions, images_names):
        fig, axis = plt.subplots()

        image = np.transpose(image, (1,2,0))
        axis.imshow(image)

        add_bounding_boxes_to_axis(labels, axis, classes, height_and_width_info, color='green')
        add_bounding_boxes_to_axis(prediction, axis, classes, height_and_width_info, color='red')

        image_path = os.path.join(output_dir, image_name)
        plt.savefig(image_path)

        #labels = labels.reshape((-1, labels.shape[-1]))
        #np.savetxt(image_path + '.txt', labels, fmt='% 1.2f')

        prediction = prediction.reshape((-1, prediction.shape[-1]))
        np.savetxt(image_path + '.txt', prediction, fmt='% 1.2f')


        