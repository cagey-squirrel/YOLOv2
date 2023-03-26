from pytorch_pipeline_util import production_dataloader
from yolo_network import TinyYOLOv2
import torch
from draw_rect import production_output
import os
from time import time


def predict(network, data, device, valid_params, mode):
    
    height_and_width_info, output_dir_path, classes, confidence_treshold = valid_params
    with torch.set_grad_enabled(False):
        for images, images_names in data:

            images = images.to(device)
            predictions = network(images)
            production_output(images, predictions, images_names, valid_params, classes, mode)
    


def production(classes, height_and_width_info, input_params):

    num_classes = len(classes)
    *rest, anchors = height_and_width_info
    num_epochs = input_params['num_epochs']

    images_dir_path = input_params['images_dir_path']
    labels_path = input_params['labels_path']

    confidence_treshold = input_params['confidence_treshold']
    overlap_treshold = input_params['overlap_treshold']
    network_type = input_params['network_type']
    augment = input_params['augment']
    mode = input_params['mode']
    
    output_dir_name = input_params['output_dir_name']
    output_dir_name += '_production_' + str(time())

    images_output_dir_name = input_params['images_output_dir_name']
    output_dir_path = os.path.join(images_output_dir_name, output_dir_name)
    os.mkdir(output_dir_path)

    print('Making datasets...')
    data = production_dataloader(images_dir_path)
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    network = TinyYOLOv2(num_classes=num_classes, anchors=anchors, network_type=network_type)

    trained_model_path = input_params['trained_model_path']
    state_dict = torch.load(trained_model_path) if torch.cuda.is_available() else torch.load(trained_model_path, map_location='cpu')
    network.load_state_dict(state_dict)
    network.to(device)
    
    valid_params = height_and_width_info, output_dir_path, classes, confidence_treshold
    predict(network, data, device, valid_params, mode)
        

       
    
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     outputs = network(images)
    #     non_max_surpression(outputs)

        #for image, output in zip(images, outputs):
        #    display_images_with_bounding_boxes(image, output, classes, 32, 32, 300, 250)
    
    return network

