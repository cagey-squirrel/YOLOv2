from pytorch_pipeline_util import make_torch_dataloaders
from yolo_network import TinyYOLOv2
from loss import YoloLoss
import torch
from draw_rect import non_max_surpression, display_images_with_bounding_boxes, output_predictions, write_metrics, average_metrics
import os
from time import time
from torch.nn.utils.clip_grad import clip_grad_value_
import numpy as np
import random
import json

def training_epoch(network, train_data, loss_function, optimizer, device, epoch_num, train_params):
    
    total_loss = 0 
    batches = 0
    *unused, train_text_file, classes, confidence_treshold, mode, overlap_treshold = train_params
    metrics = np.array([[0, 0] for _ in classes])
    tp_fp_fn = np.array([[0, 0, 0] for _ in classes])

    with torch.set_grad_enabled(True):
        optimizer.zero_grad()
        for (images, images_names), labels in train_data:
            images, labels = images.to(device), labels.to(device)
            predictions = network(images)

            loss = loss_function(predictions, labels)
            loss.backward()

            total_loss += loss.item()

            optimizer.step()

            if epoch_num % 10 == 0: #or epoch_num < 10:
                new_metrics, new_tp_fp_fn = output_predictions(images, labels, predictions, images_names, epoch_num, train_params, classes, batches)
                metrics += new_metrics
                tp_fp_fn += new_tp_fp_fn
            
            batches += 1
    
    if epoch_num % 10 == 0: # or epoch_num < 10:
        averaged_metrics = average_metrics(metrics)
        write_metrics(averaged_metrics, tp_fp_fn, classes, train_text_file, epoch_num)        
        
            

    total_loss /= batches 

    
    print(f'epoch {epoch_num} train loss = {total_loss}')
    #input('train_loss')
    return total_loss

def validation_epoch(network, validation_data, loss_function, device, epoch_num, valid_params):
    
    total_loss = 0 
    batches = 0
    *unused, valid_text_file, classes, confidence_treshold, mode, overlap_treshold = valid_params
    metrics = np.array([[0, 0] for _ in classes])

    with torch.set_grad_enabled(False):
        for (images, images_names), labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            predictions = network(images)
            loss = loss_function(predictions, labels)
            #loss = 0
            total_loss += loss
            

            if epoch_num % 10 == 0: # or epoch_num < 10:
                new_metrics, new_tp_fp_fn = output_predictions(images, labels, predictions, images_names, epoch_num, valid_params, classes, batches)
                metrics += new_metrics
                tp_fp_fn += new_tp_fp_fn
            
            batches += 1
    
    if epoch_num % 10 == 0: # or epoch_num < 10:
        averaged_metrics = average_metrics(metrics)
        write_metrics(averaged_metrics, tp_fp_fn, classes, valid_text_file, epoch_num)

            

    total_loss /= batches 

    print(f'epoch {epoch_num} valid loss = {total_loss}')
    return total_loss

def training(classes, height_and_width_info, input_params):

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
    output_dir_name += str(time())

    images_output_dir_name = input_params['images_output_dir_name']
    output_dir_path = os.path.join(images_output_dir_name, output_dir_name)
    os.mkdir(output_dir_path)

    trained_models_output_dir_name = input_params['trained_models_output_dir_name']
    trained_models_dir_path = os.path.join(trained_models_output_dir_name, output_dir_name)
    os.mkdir(trained_models_dir_path)

    train_output_dir_path = os.path.join(output_dir_path, 'train')
    valid_output_dir_path = os.path.join(output_dir_path, 'valid')
    os.mkdir(train_output_dir_path)
    os.mkdir(valid_output_dir_path)
    train_metrics_path = os.path.join(train_output_dir_path, 'train.txt')
    valid_metrics_path = os.path.join(valid_output_dir_path, 'valid.txt')
    train_text_file = open(train_metrics_path, 'a+')
    valid_text_file = open(valid_metrics_path, 'a+')
    train_text_file.writelines(json.dumps(input_params) + '\n')
    valid_text_file.writelines(json.dumps(input_params) + '\n')

    print('Making datasets...')
    train_loader, test_loader = make_torch_dataloaders(images_dir_path, labels_path, classes, height_and_width_info, augment=augment)
    
    loss_function = YoloLoss(input_params)
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    network = TinyYOLOv2(num_classes=num_classes, anchors=anchors, network_type=network_type)

    if input_params['overtrain_model']:
        trained_model_path = input_params['trained_model_path']
        state_dict = torch.load(trained_model_path) if torch.cuda.is_available() else torch.load(trained_model_path, map_location='cpu')
        network.load_state_dict(state_dict)
        
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=input_params['learning_rate'])
    clip_grad_value_(network.parameters(), input_params['clip_gradient_value'])

    train_params = height_and_width_info, train_output_dir_path, train_text_file, classes, confidence_treshold, mode, overlap_treshold
    valid_params = height_and_width_info, valid_output_dir_path, valid_text_file, classes, confidence_treshold, mode, overlap_treshold

    print('Starting training...')
    for epoch_num in range(num_epochs):
        
        time_start_epoch = time()
        validation_epoch(network, test_loader, loss_function, device, epoch_num, valid_params)
        training_epoch(network, train_loader, loss_function, optimizer, device, epoch_num, train_params)
        print(f'epoch {epoch_num} finished in {time() - time_start_epoch}\n')

        if (epoch_num + 1) % 50 == 0:
            torch.save(network.state_dict(), os.path.join(trained_models_dir_path, f"unet_model__{(epoch_num + 1)}.pt"))
    
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     outputs = network(images)
    #     non_max_surpression(outputs)

        #for image, output in zip(images, outputs):
        #    display_images_with_bounding_boxes(image, output, classes, 32, 32, 300, 250)
    
    return network
    