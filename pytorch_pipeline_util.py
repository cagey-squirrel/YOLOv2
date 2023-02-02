
import os 
import torch 
import numpy as np
from PIL import Image
from collections import defaultdict
from math import floor
from torch.utils.data import random_split, Dataset, DataLoader
from draw_rect import display_images_with_bounding_boxes


def get_center_and_size_of_annotation(annotation, cell_height, cell_width, anchor_height, anchor_width):
    '''
    Calculates objects center coordinate and the size of bounding box
    Center coordinates are divided by cell sizes
    Bounding box sizes are divided by anchor sizes
    '''
    top_left_corner_y, top_left_corner_x, bot_right_corner_y, bot_right_corner_x, label = annotation
    box_center_y = (bot_right_corner_y + top_left_corner_y) // 2
    box_center_x = (bot_right_corner_x + top_left_corner_x) // 2 

    box_height = bot_right_corner_y - top_left_corner_y
    box_width = bot_right_corner_x - top_left_corner_x

    
    box_center_y /= cell_height
    box_center_x /= cell_width

    box_height /= anchor_height
    box_width /= anchor_width

    return box_center_x, box_center_y, box_width, box_height, label


def load_labels(annotations_path, image_names, image_height, image_width, anchor_height, anchor_width, dimension_reduction):
    '''
    Loads labels from csv file on annotations_path

    Input:
        - annotations_path (string): Path to csv file containing annotations. More details in load_annotations function.
        - image_names (list of strings): labels need to be loaded in the same order as images thet they represent
        - image_height (int): height of the image in pixels
        - image_width (int): width of the image in pixels
        - anchor_height (int): height of the anchor in pixels
        - anchor_width (int): width of the anchor in pixels


    Returns:
        - labels (dict): dict containing label (Tensor) for each image
          Each image has W * H cells, each cell has a label of length equal to (5 + Num_Classes)
          So for each image, label has a shape of (W, H, (5 + Num_Classes))
          (5 + NumClasses) represents [AnnotationCenterX, AnnotationCenterY, AnnotationWidth, AnnotationHeight, Confidence, Class1_Prob, Class2_prob...]
    
    '''
    
    cell_height = dimension_reduction
    cell_width = dimension_reduction

    num_cells_height = image_height // cell_height
    num_cells_width = image_width // cell_width

    annotations = load_annotations(annotations_path)
    #labels = defaultdict(lambda: [])
    labels = []

    # Getting indexes of labels
    classes = ['Alan', 'Jeremija', 'Brok', 'Broj 1', 'Sir Oliver', 'Grunt']
    number_of_classes = len(classes)
    agent_name_to_class_number = {}
    for i, agent_name in enumerate(classes):
        agent_name_to_class_number[agent_name] = i
    
    # Shape of a single label is (5 + NumClasses)
    # label = [box_center_x, box_center_y, box_width, box_height, object_is_in_this_cell, one_hot_class_encodings]
    for image_name in image_names:
        
        # Each image has a label for each cell so the shape of image_labels is (Num_cells_width, Num_cells_height, (5 + NumClasses))
        image_labels = np.zeros(shape=(num_cells_width, num_cells_height, (5 + number_of_classes)))

        annotation_list = annotations[image_name]
        
        for annotation in annotation_list:
            box_center_x, box_center_y, box_width, box_height, class_name = get_center_and_size_of_annotation(annotation, cell_height, cell_width, anchor_height, anchor_width)
            
            box_center_cell_index_x = floor(box_center_x)
            box_center_cell_index_y = floor(box_center_y)

            label = [box_center_x, box_center_y, box_width, box_height]
            label.append(1) # object is in this cell -> certainty is 1
            class_one_hot_encoding = [0 for _ in range(number_of_classes)]
            class_one_hot_encoding[agent_name_to_class_number[class_name]] = 1
            label.extend(class_one_hot_encoding)
            
            label = np.array(label)
            image_labels[box_center_cell_index_x, box_center_cell_index_y, :] = label

        #print(image_labels)
        #input('labels')
        image_labels = torch.Tensor(image_labels)
        labels.append(image_labels)
        #labels[image_name] = image_labels
    
    return labels


def load_annotations(annotations_path):
    '''
    Loads annotations from annotations_path

    Input:
        - annotations_path (string): path to .csv file which contains annotations
            csv file should have format of: image_name, object_center_x, object_center_y, object_width, object_height, object_label
            image_name and object_label are strings, all other values are float and meassured in pixels
    
    Returns:
        - annotations (dict): dict containing image name as key. Dict value is the list which contains every object detection for given image name
          since images can have multiple object on them each image has a list of object in this dict
    '''
    
    annotation_file = open(annotations_path, "r")
    lines = annotation_file.read().splitlines()

    annotations = defaultdict(lambda: [])
    for line in lines[1:]: # Skipping first line which has column names
        image_info = line.split(',')
        image_name = (image_info[0].strip('\"'))[:-4] # removing quotes from string and removing '.png' ext from name
        top_left_corner_x = int(float(image_info[1]))
        top_left_corner_y = int(float(image_info[2]))
        bot_right_corner_x = int(float(image_info[3]))
        bot_right_corner_y = int(float(image_info[4]))
        label = image_info[5].strip('\"')

        annotations[image_name].append((top_left_corner_y, top_left_corner_x, bot_right_corner_y, bot_right_corner_x, label))
    
    return annotations


def load_images(images_dir):
    '''
    Loads images from image_dir and returnes them as tensors

    Input:
        - images_dir (string): path to directory which contains images
    
    Returns:
        - images (dict): dict containing image names as keys and tensors as images
            example for loading images 'alan.jpg' : images['alan'] = Tensor(1, 4, 256, 256) where 4 is the num of channels
    '''
    
    images = []
    image_names = []

    for image_name in os.listdir(images_dir):

        image_path = os.path.join(images_dir, image_name)
        image = np.array(Image.open(image_path))
        image = torch.Tensor(image)
        image = torch.permute(image, (2, 0, 1))
        #image = image[None, :, :, :]
        
        image_id = image_name[:-4] # removing '.png' extension from image
        images.append(image)
        image_names.append(image_id)
        # images[image_id] = image  
    
    image_height, image_width = image.shape[1], image.shape[2]

    return images, image_names, image_height, image_width


def load_images_and_labels(images_dir_path, labels_path, anchor_height, anchor_width, dimension_reduction):
    '''
    Loads images and labels

    Input:
        - images_path (string): path to directory which contains images
        - labels_path (string): path to a csv file which contains labels:
        - dimension_reduction (int): when image passes through CNN its dimensions reduce in each layer
          This number is equal to image_size_before_CNN // image_size_after_CNN
    
    Returns:
        - image_data: list which contains images
        - labels_data: list which contains labels
    '''
    
    images_data, image_names, image_height, image_width = load_images(images_dir_path)
    #print(f'loadimglbl image_height = {image_height}, image_width = {image_width}')
    labels_data = load_labels(labels_path, image_names, image_height, image_width, anchor_height, anchor_width, dimension_reduction)

    image_height = 384
    image_width = 576
    num_cells_height = 12 
    num_cells_width = 18
    anchor_height = 250
    anchor_width = 300
    cell_width = 32
    cell_height = 32

    return images_data, labels_data

def train_test_split(images, labels, test_percentage):
    '''
    Splits images and labels into train and test dataset based on test_percentage.
    Transforms data into tensors

    Input:
        - images (list): list of loaded images
        - labels (list): list of loaded labels (in same order as images)
        - test_percentage (int): percentage of examples that will go to test dataset (others will go to training)
    
    Returns:
        - train_images (torch.Tensor): images for training 
        - train_labels (torch.Tensor): labels for training
        - test_images (torch.Tensor): images for testing
        - test_labels (torch.Tensor): labels for testing
    '''

    total_length = len(images)
    test_data_len = int(total_length * test_percentage / 100)

    images = torch.stack(images)
    labels = torch.stack(labels)


    # Shuffleling data
    random_permutation = torch.randperm(total_length)
    images = images[random_permutation]
    labels = labels[random_permutation]

    test_images = images[:test_data_len]
    test_labels = labels[:test_data_len]
    train_images = images[test_data_len:]
    train_labels = labels[test_data_len:]
    
    return train_images, train_labels, test_images, test_labels


class YoloDetectionDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        return img, label
    

def make_torch_datasets(training_images, training_labels, test_images, test_labels):
    '''
    Wraps datasets in torch.utils.data.Dataset class and returns training and testing dataset
    '''
    train_dataset = YoloDetectionDataset(training_images, training_labels)
    test_dataset = YoloDetectionDataset(test_images, test_labels)

    return train_dataset, test_dataset


def make_torch_dataloaders(images_dir_path, labels_path, batch_size=8):
    '''
    This function loads all images and labels, splits them randomly into training and test set and wraps these datasets in DataLoader class.
    '''
    dimension_reduction = 32
    anchor_height = 250
    anchor_width = 300

    images, labels = load_images_and_labels(images_dir_path, labels_path, anchor_height, anchor_width, dimension_reduction)
    training_images, training_labels, test_images, test_labels = train_test_split(images, labels, test_percentage=10)
    train_dataset, test_dataset = make_torch_datasets(training_images, training_labels, test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(1302))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=torch.Generator().manual_seed(1302))

    return train_loader, test_loader


def main():
    print('test')
    images_dir_path = 'all_same_size_imgs'
    labels_path = 'annotations/annotations2.csv'
    anchor_height = 250
    anchor_width = 300
    dimension_reduction = 32
    images, labels = load_images_and_labels(images_dir_path, labels_path, anchor_height, anchor_width, dimension_reduction)

    train_images, train_labels, test_images, test_labels = train_test_split(images, labels, test_percentage=10)

    for image, label in zip(train_images, train_labels):
        display_images_with_bounding_boxes(image, label, 32, 32, anchor_width, anchor_height)

    print(f'train_images.shape = {train_images.shape}')
    print(f'train_labels.shape = {train_labels.shape}')
    print(f'test_images.shape = {test_images.shape}')
    print(f'test_labels.shape = {test_labels.shape}')

if __name__ == "__main__":
    main()