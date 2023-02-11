from pytorch_pipeline_util import make_torch_dataloaders
from yolo_network import TinyYOLOv2
from loss import YoloLoss
import torch
from draw_rect import non_max_surpression, display_images_with_bounding_boxes, output_predictions
import os
from time import time

def training_epoch(network, train_data, loss_function, optimizer, device, do_output_predictions, height_and_width_info, epoch_num, train_output_dir_path):
    
    total_loss = 0 
    batches = 0

    with torch.set_grad_enabled(True):
        optimizer.zero_grad()
        for (images, images_names), labels in train_data:
            images, labels = images.to(device), labels.to(device)
            predictions = network(images)

            loss = loss_function(predictions, labels)
            loss.backward()

            total_loss += loss.item()
            batches += 1


            optimizer.step()

    total_loss /= batches 

    if do_output_predictions:
        non_max_surpression(predictions)
        output_predictions(images, labels, predictions, images_names, classes, height_and_width_info, train_output_dir_path, epoch_num)

    return total_loss

def validation_epoch(network, validation_data, loss_function, device, do_output_predictions, height_and_width_info, epoch_num, valid_output_dir_path):
    
    total_loss = 0 
    batches = 0

    with torch.set_grad_enabled(False):
        for (images, images_names), labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            predictions = network(images)
            loss = loss_function(predictions, labels)
            total_loss += loss
            batches += 1

    total_loss /= batches 

    if do_output_predictions:
        non_max_surpression(predictions)
        output_predictions(images, labels, predictions, images_names, classes, height_and_width_info, valid_output_dir_path, epoch_num)

    return total_loss

def training(images_dir_path, annotations_path, classes, num_epochs, height_and_width_info, output_dir_path):

    num_classes = len(classes)
    anchors = [(250, 300)]

    train_output_dir_path = os.path.join(output_dir_path, 'train')
    valid_output_dir_path = os.path.join(output_dir_path, 'valid')
    os.mkdir(train_output_dir_path)
    os.mkdir(valid_output_dir_path)

    train_loader, test_loader = make_torch_dataloaders(images_dir_path, annotations_path, classes)
    network = TinyYOLOv2(num_classes=num_classes, anchors=anchors)
    loss_function = YoloLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    network.to(device)
    
    for epoch_num in range(num_epochs):

        do_output_predictions = (epoch_num % 10 == 0)
        output_predictions_params = height_and_width_info, epoch_num
        
        val_loss = validation_epoch(network, test_loader, loss_function, device, do_output_predictions, height_and_width_info, epoch_num, valid_output_dir_path)
        train_loss = training_epoch(network, train_loader, loss_function, optimizer, device, do_output_predictions, height_and_width_info, epoch_num, train_output_dir_path)

        print(f'val_loss = {val_loss}')
        print(f'train_loss = {train_loss}')
        print(f'{epoch_num}/{num_epochs}')
    
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     outputs = network(images)
    #     non_max_surpression(outputs)

        #for image, output in zip(images, outputs):
        #    display_images_with_bounding_boxes(image, output, classes, 32, 32, 300, 250)
    
    return network
    


if __name__ == "__main__":
    images_dir_path = 'C:\\Users\\cvetk\\Desktop\\small_data\\images'
    labels_path = 'C:\\Users\\cvetk\\Desktop\\small_data\\annotations\\annotations.csv'
    classes = ['Broj 1', 'Alan Ford', 'Bob Rok', 'Sir Oliver', 'Grunf', 'Jeremija', 'Sef']

    output_dir_name = 'first_training2' + str(time())
    output_dir_path = os.path.join('output', output_dir_name)
    os.mkdir(output_dir_path)

    image_height = 384
    image_width = 576
    num_cells_height = 12 
    num_cells_width = 18
    anchor_height = 250
    anchor_width = 300
    cell_width = 32
    cell_height = 32

    
    height_and_width_info = image_height, image_width, cell_width, cell_height, anchor_width, anchor_height
    
    training(images_dir_path, labels_path, classes, 100, height_and_width_info, output_dir_path)
    


    



    
