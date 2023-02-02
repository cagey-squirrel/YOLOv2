from pytorch_pipeline_util import make_torch_dataloaders
from yolo_network import TinyYOLOv2
from loss import YoloLoss
import torch

def training_epoch(network, train_data, loss_function, optimizer, device):
    
    total_loss = 0 
    batches = 0

    with torch.set_grad_enabled(True):
        optimizer.zero_grad()
        for images, labels in train_data:

            images, labels = images.to(device), labels.to(device)
            predictions = network(images)

            loss = loss_function(predictions, labels)
            loss.backward()

            total_loss += loss.item()
            batches += 1


            optimizer.step()

    total_loss /= batches 

    return total_loss

def validation_epoch(network, validation_data, loss_function, device):
    
    total_loss = 0 
    batches = 0

    with torch.set_grad_enabled(False):
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            predictions = network(images)
            loss = loss_function(predictions, labels)
            total_loss += loss
            batches += 1

    total_loss /= batches 

    return total_loss

def training(images_dir_path, annotations_path, num_epochs):

    classes = ['Alan', 'Jeremija', 'Brok', 'Broj 1', 'Sir Oliver', 'Grunt']
    num_classes = len(classes)
    anchors = [(250, 300)]

    train_loader, test_loader = make_torch_dataloaders(images_dir_path, annotations_path)
    network = TinyYOLOv2(num_classes=num_classes, anchors=anchors)
    loss_function = YoloLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    network.to(device)
    
    for epoch in range(num_epochs):
        val_loss = validation_epoch(network, test_loader, loss_function, device)
        train_loss = training_epoch(network, train_loader, loss_function, optimizer, device)

        print(f'val_loss = {val_loss}')
        print(f'train_loss = {train_loss}')
    
    return network
    


if __name__ == "__main__":
    images_dir_path = 'all_same_size_imgs'
    labels_path = 'annotations/annotations2.csv'
    training(images_dir_path, labels_path, 10)
    


    



    
