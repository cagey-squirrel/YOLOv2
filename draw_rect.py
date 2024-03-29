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
from metrics.utils import *
from metrics.BoundingBox import BoundingBox
from metrics.BoundingBoxes import BoundingBoxes
from metrics.Evaluator import *


def average_metrics(metrics):
    '''
    This function calculated average metrics
    metric_pair is consisted of two numbers: metric_sum and class_present_sum
    - metric_sum is the sum of all metrics in all batches
    - class_present_sum stores how many batches actually contained this class
      this number is essential because if batch doesnt contain a certain class
      then metrics for that class will be 0, even though we didnt really make any mistake
      there's just not an object present
    '''
    averaged_metrics = []
    for metric_sum, class_present_sum in metrics:
        averaged_metrics.append(metric_sum / class_present_sum)
    
    return np.array(averaged_metrics)


def non_max_surpression(predictions, confidence_treshold=0.5):
    '''
    Surpress any predictions with confidence less than confidence_treshold
    In this task there is a maximum of one detection per class so we only take the prediction with maximum confidence per class

    Parameters:
        - predictions (torch.Tensor): prediction vector which will be surpressed
            this vector has a shape of (A, W, H, (5 + Num_Classes))
        - confidence_treshold (float: between 0 and 1): minimum confidence required for prediction
            all predictions with lower confidence than this will be surpressed
    '''

    zeros_vector = np.zeros((predictions.shape[-1]))  # shape is (5 + num_classes)
    confidences = predictions[..., 4]
    # Surpressing all predictions with confidence less than confidence_treshold
    #predictions[confidences < confidence_treshold] = zeros_vector

    class_probabilities = predictions[..., 5:]
    maxes = class_probabilities.max(axis=-1)
    maxes = maxes[..., None]
    class_one_hot = (class_probabilities == maxes)
    
    num_classes = class_probabilities.shape[-1]
    for class_index in range(num_classes):
        same_class = class_one_hot[..., class_index]
        conf_times_class = confidences * same_class
        class_conf_maxes = conf_times_class.max()
        

        predictions[np.logical_and(conf_times_class > 0, conf_times_class != class_conf_maxes)] = zeros_vector
    

def get_corners(prediction):
    '''
    Calculates top left corner and bottom right corner from prediction
    prediction is bounding box represented as [bb_center_x, bb_center_y, bb_width, bb_height, blabla]
    '''

    bb_center_x, bb_center_y, bb_width, bb_height, *rest_unused = prediction

    top_left_corner_x = bb_center_x - bb_width/2
    top_left_corner_y = bb_center_y - bb_height/2
    top_left_corner = (top_left_corner_x, top_left_corner_y)

    bot_right_corner_x = bb_center_x + bb_width/2
    bot_right_corner_y = bb_center_y + bb_height/2
    bot_right_corner = (bot_right_corner_x, bot_right_corner_y)

    return top_left_corner, bot_right_corner


def get_intersect(top_left_corner1, bot_right_corner_1, top_left_corner2, bot_right_corner_2):
    '''
    Returns the area of intersection between two boxes bounded by inputs
    '''
    
    intersect_width = min(bot_right_corner_1[0], bot_right_corner_2[0]) - max(top_left_corner1[0], top_left_corner2[0])
    intersect_height = min(bot_right_corner_1[1], bot_right_corner_2[1]) - max(top_left_corner1[1], top_left_corner2[1])

    return intersect_width * intersect_height




def get_intersection_over_union(prediction1, prediction2):
    '''
    Returns intersection over union value between two bounding boxes represented by prediction1 and prediction2
    '''
    top_left_corner1, bot_right_corner_1 = get_corners(prediction1)
    top_left_corner2, bot_right_corner_2 = get_corners(prediction2)

    # Calculating areas of bounding boxes: area = (brc.y - tlc.y) * (brc.x - tlc.x)
    area1 = (bot_right_corner_1[0] - top_left_corner1[0]) * (bot_right_corner_1[1] - top_left_corner1[1])
    area2 = (bot_right_corner_2[0] - top_left_corner2[0]) * (bot_right_corner_2[1] - top_left_corner2[1])

    intersect = get_intersect(top_left_corner1, bot_right_corner_1, top_left_corner2, bot_right_corner_2)

    return intersect / (area1 + area2 - intersect)




def surpress_overlaping_detections(predictions, overlap_treshold=0.5):
    '''
    Returns a list of filtered predictions.
    If a prediction overlaps with another prediction for more than overlap_tereshold then the prediction with lower confidence is filtered
    '''

    active_predictions = {}
    key_iter = 0

    for prediction in predictions:
        confidence = prediction[4]
        keep_prediction = True
        ids_for_removal = []
        if confidence > 0:
            for key in active_predictions:
                active_prediction = active_predictions[key]
                iou = get_intersection_over_union(prediction, active_prediction)
                if iou > overlap_treshold:
                    active_prediction_confidence = active_prediction[4]
                    if active_prediction_confidence > confidence: # Current prediction overlaps above treshold with another prediction with higher confidence
                        keep_prediction = False
                        break # So we do not save current prediction
                    else: # This prediction has a higher confidence thatn active prediction it overlaps with
                        # So we delete that active prediction
                        ids_for_removal.append(key)
            if keep_prediction:
                active_predictions[key_iter] = prediction
                key_iter += 1

                for id_for_removal in ids_for_removal:
                    del active_predictions[id_for_removal]
    
    # print(f'len = {len(list(active_predictions.values()))}')
    return list(active_predictions.values())

            

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


def add_bounding_boxes_to_axis_and_bounding_boxes(bounding_boxes, axis, bounding_boxes_list, classes, image_name, ground_truth, color, confidence_treshold, mode):
    '''
    Adds bounding box to axis so it can be plotted.
    Adds bounding box to list of bounding boxes so we can track its metrics
    '''


    non_max_surpression(bounding_boxes)
    bounding_boxes = bounding_boxes.reshape(-1, bounding_boxes.shape[-1])
    bounding_boxes = surpress_overlaping_detections(bounding_boxes)
    
    for bounding_box in bounding_boxes:


        if bounding_box[4] < confidence_treshold:
            continue

        # Bounding box contains: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
        box_center_x = bounding_box[0]
        box_center_y = bounding_box[1]

        box_width  = bounding_box[2]
        box_height = bounding_box[3]

        
        # Class Rectangle needs top left corner as input
        top_left_corner_x = box_center_x - box_width  / 2
        top_left_corner_y = box_center_y - box_height / 2

        class_probabilities = bounding_box[5:]
        class_max_index = class_probabilities.argmax()
        object_class = classes[class_max_index]
        confidence = bounding_box[4]

        class_and_confidence = object_class + " " + str(confidence*100)[:3] + "%"

        # Making sure the box fits the image (doesnt go beyond)
        # top_left_corner_x = min(max(1, top_left_corner_x), image_width-1)
        # top_left_corner_y = min(max(1, top_left_corner_y), image_height-1)
        
        rect = matplotlib.patches.Rectangle((top_left_corner_x, top_left_corner_y), box_width, box_height, fill=False, edgecolor=color) 
        axis.text(x=top_left_corner_x + 10, y = top_left_corner_y + 20, s = class_and_confidence, color = color)
        axis.add_patch(rect)

        if mode != 'production':
            bbType = BBType.GroundTruth if ground_truth else BBType.Detected
            bb = BoundingBox(image_name, object_class, top_left_corner_x, top_left_corner_y, box_width, box_height, bbType=bbType, classConfidence=confidence)
            bounding_boxes_list.addBoundingBox(bb)


def output_predictions(images, labels_list, predictions, images_names, epoch_num, params, classes, batch_num):
    '''
    Outputs prediction detections and true label boxes on images

    Input:
        - images:
        - labels
        - predictions
        - images_names
    '''

    height_and_width_info, output_dir_path, text_file, classes, confidence_treshold, mode, overlap_treshold = params

    output_dir = os.path.join(output_dir_path, str(epoch_num))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    images = images.detach().cpu().numpy()
    labels_list = labels_list.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    image_height, image_width, cell_width, cell_height, anchors = height_and_width_info 
    # Bounding box contains: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
    # Box_center_x, and Box_center_y are in units of cells so we need to multiply them to display them on image
    # box_width and box_height are in units of anchor sizes so they too need to be multiplied in order to be displayed
    #print(f'box_center_x = {bounding_box[0]} box_center_y = {bounding_box[1]}')

    labels_list[..., 0] *= cell_width
    labels_list[..., 1] *= cell_height
    labels_list[..., 2] *= anchors[:,0]
    labels_list[..., 3] *= anchors[:,1]

    predictions[..., 0] *= cell_width
    predictions[..., 1] *= cell_height
    predictions[..., 2] *= anchors[:,0]
    predictions[..., 3] *= anchors[:,1]

    bounding_boxes_list = BoundingBoxes()
    evaluator = Evaluator()

    fig, axis = plt.subplots()
    #axis = None
    for image, labels, prediction, image_name in zip(images, labels_list, predictions, images_names):

        add_bounding_boxes_to_axis_and_bounding_boxes(labels, axis, bounding_boxes_list, classes, image_name, ground_truth=True, color='green', confidence_treshold=confidence_treshold, mode=mode)
        add_bounding_boxes_to_axis_and_bounding_boxes(prediction, axis, bounding_boxes_list, classes, image_name, ground_truth=False, color='red', confidence_treshold=confidence_treshold, mode=mode)

        if batch_num < 3 or (mode == 'testing' and epoch_num==1):
            image = np.transpose(image, (1,2,0))
            axis.imshow(image)
            image_path = os.path.join(output_dir, image_name)
            plt.savefig(image_path)
            plt.cla()
            prediction = prediction.reshape((-1, prediction.shape[-1]))
            np.savetxt(image_path + '.txt', prediction, fmt='% 1.2f')

        #labels = labels.reshape((-1, labels.shape[-1]))
        #np.savetxt(image_path + '.txt', labels, fmt='% 1.2f')
    
    plt.close()

        
    
    
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        bounding_boxes_list,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=overlap_treshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation # As the official matlab code
        ) 
    
    
   
    # text_file.write(f'epoch {epoch_num} ')
    # Loop through classes to obtain their metrics
    total_average_precision = 0
    
    metrics_dict = {}
    tp_fp_fn_dict = {}

    metricsPerClass = sorted(metricsPerClass, key=lambda x: x['class'])
    for mc in metricsPerClass:
        c = mc['class']
        average_precision = mc['AP']
        total_average_precision += average_precision
        #ipre = mc['interpolated precision']
        #irec = mc['interpolated recall']
        #precision = mc['precision']
        #recall = mc['recall']
        metrics_dict[c] = average_precision
        tp, fp, fn = mc['TP'], mc['FP'], mc['FN']
        tp_fp_fn_dict[c] = [tp, fp, fn]
        # text_file.write(f'{c} = {str(average_precision)[:4]}    ')
    

    
    metrics = np.array([[0, 0] for _ in classes])
    tp_fp_fn = np.array([[0, 0, 0] for _ in classes]).astype('int64')

    for i, object_class in enumerate(classes):
        if object_class in metrics_dict:
            metrics[i][0] = metrics_dict[object_class]
            metrics[i][1] = 1

            current_tp_fp_fn = np.array(tp_fp_fn_dict[object_class]).astype('int64')
            tp_fp_fn[i] = current_tp_fp_fn


    # total_average_precision /= len(metricsPerClass)
    # text_file.write(f'average = {str(total_average_precision)[:4]}')

    # text_file.write('\n')
    return metrics, tp_fp_fn


def production_output(images, predictions, images_names, params, classes, mode):

    height_and_width_info, output_dir, classes, confidence_treshold = params
    
    images = images.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    image_height, image_width, cell_width, cell_height, anchors = height_and_width_info 
    # Bounding box contains: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
    # Box_center_x, and Box_center_y are in units of cells so we need to multiply them to display them on image
    # box_width and box_height are in units of anchor sizes so they too need to be multiplied in order to be displayed
    #print(f'box_center_x = {bounding_box[0]} box_center_y = {bounding_box[1]}')

    predictions[..., 0] *= cell_width
    predictions[..., 1] *= cell_height
    predictions[..., 2] *= anchors[:,0]
    predictions[..., 3] *= anchors[:,1]

    

    fig, axis = plt.subplots()
    #axis = None
    for image, prediction, image_name in zip(images, predictions, images_names):

        add_bounding_boxes_to_axis_and_bounding_boxes(prediction, axis, None, classes, image_name, ground_truth=False, color='red', confidence_treshold=confidence_treshold, mode=mode)

        image = np.transpose(image, (1,2,0))
        axis.imshow(image)
        image_path = os.path.join(output_dir, image_name)
        plt.savefig(image_path)
        plt.cla()
        prediction = prediction.reshape((-1, prediction.shape[-1]))
        np.savetxt(image_path + '.txt', prediction, fmt='% 1.2f')

        #labels = labels.reshape((-1, labels.shape[-1]))
        #np.savetxt(image_path + '.txt', labels, fmt='% 1.2f')
    
    plt.close()


def write_metrics(metrics, tp_fp_fn, classes, text_file, epoch_num):

    text_file.write(f'epoch: = {epoch_num}   ')

    for metric, metric_class in zip(metrics, classes):
        text_file.write(f'{metric_class} = {str(metric)[:5]}   ')
    
    mean_metric = metrics.mean()
    text_file.write(f'average = {str(mean_metric)[:5]}\n')

    for (tp, fp, fn), metric_class in zip(tp_fp_fn, classes):
        text_file.write(f'{metric_class} : TP:{tp} FP:{fp} FN:{fn}  ')
    
    total = tp_fp_fn.sum(axis=0)

    text_file.write(f'total: TP:{total[0]} FP:{total[1]} FN:{total[2]}')    

    text_file.write('\n\n')

    text_file.flush()
    


        