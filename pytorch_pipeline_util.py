
import os 
import torch 
import numpy as np
from PIL import Image
from collections import defaultdict
from math import floor
from torch.utils.data import random_split, Dataset, DataLoader
from draw_rect import display_images_with_bounding_boxes
import random
from matplotlib import pyplot as plt
from augmentors import augment_image, flip_image_and_annotation


def get_center_and_size_of_annotation(annotation, cell_height, cell_width, anchors):
    '''
    Calculates objects center coordinate and the size of bounding box
    Center coordinates are divided by cell sizes
    Bounding box sizes are divided by anchor sizes
    '''
    top_left_corner_y, top_left_corner_x, bot_right_corner_y, bot_right_corner_x, label, anchor_index = annotation
    box_center_y = (bot_right_corner_y + top_left_corner_y) // 2
    box_center_x = (bot_right_corner_x + top_left_corner_x) // 2 

    box_height = bot_right_corner_y - top_left_corner_y
    box_width = bot_right_corner_x - top_left_corner_x
    
    box_center_y /= cell_height
    box_center_x /= cell_width

    anchor_height, anchor_width = anchors[anchor_index]
    box_height /= anchor_height
    box_width /= anchor_width

    return box_center_x, box_center_y, box_width, box_height, label, anchor_index


def load_labels(annotations_path, image_names, classes, height_and_width_info):
    '''
    Loads labels from csv file on annotations_path

    Input:
        - annotations_path (string): Path to csv file containing annotations. More details in load_annotations function.
        - image_names (list of strings): labels need to be loaded in the same order as images thet they represent
        - image_height (int): height of the image in pixels
        - image_width (int): width of the image in pixels
        - anchor_height (int): height of the anchor in pixels
        - anchor_width (int): width of the anchor in pixels
        - dimension_reduction (int): image_size_before_network / image_size_after_network
        - classes (list): list of all classes in dataset (example: ['passanger', 'car', 'tree'])


    Returns:
        - labels (list): dict containing label (Tensor) for each image
          Each image has W * H cells, each cell has a label of length equal to (5 + Num_Classes)
          So for each image, label has a shape of (W, H, (5 + Num_Classes))
          (5 + NumClasses) represents [AnnotationCenterX, AnnotationCenterY, AnnotationWidth, AnnotationHeight, Confidence, Class1_Prob, Class2_prob...]
    
    '''

    image_height, image_width, cell_width, cell_height, anchors = height_and_width_info
    num_anchors = anchors.shape[0]

    num_cells_height = image_height // cell_height
    num_cells_width = image_width // cell_width


    annotations = load_annotations(annotations_path, anchors)
    #labels = defaultdict(lambda: [])
    labels = []

    # Getting indexes of labels
    number_of_classes = len(classes)
    agent_name_to_class_number = {}
    for i, agent_name in enumerate(classes):
        agent_name_to_class_number[agent_name] = i
    
    # Shape of a single label is (5 + NumClasses)
    # label = [box_center_x, box_center_y, box_width, box_height, object_is_in_this_cell, one_hot_class_encodings]
    for image_name in image_names:
        
        # Each image has a label for each cell so the shape of image_labels is (Num_cells_width, Num_cells_height, (5 + NumClasses))
        image_labels = np.zeros(shape=(num_cells_width, num_cells_height, num_anchors, (5 + number_of_classes)))

        annotation_list = annotations[image_name]
        
        for annotation in annotation_list:

            box_center_x, box_center_y, box_width, box_height, class_name, anchor_index = get_center_and_size_of_annotation(annotation, cell_height, cell_width, anchors)
      
            box_center_cell_index_x = floor(box_center_x)
            box_center_cell_index_y = floor(box_center_y)

            label = [box_center_x, box_center_y, box_width, box_height]
            label.append(1) # object is in this cell -> certainty is 1
            class_one_hot_encoding = [0 for _ in range(number_of_classes)]
            class_one_hot_encoding[agent_name_to_class_number[class_name]] = 1
            label.extend(class_one_hot_encoding)
            
            label = np.array(label)
            image_labels[box_center_cell_index_x, box_center_cell_index_y, anchor_index, :] = label

        image_labels = torch.Tensor(image_labels)
        labels.append(image_labels)
    
    return labels

def get_anchor_index(top_left_corner_x, top_left_corner_y, bot_right_corner_x, bot_right_corner_y, anchors):

    width = bot_right_corner_x - top_left_corner_x
    height = bot_right_corner_y - top_left_corner_y

    if height < 200 and width < 200:
        anchor_index = 0
    else:
        anchor_index = 1
    
    # see as you can see
    #return 0
    return anchor_index


def load_annotations(annotations_path, anchors):
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

        anchor_index = get_anchor_index(top_left_corner_x, top_left_corner_y, bot_right_corner_x, bot_right_corner_y, anchors)

        annotations[image_name].append((top_left_corner_y, top_left_corner_x, bot_right_corner_y, bot_right_corner_x, label, anchor_index))
    
    return annotations


def load_images(images_dir, normalize=True):
    '''
    Loads images from image_dir

    Input:
        - images_dir (string): path to directory which contains images
    
    Returns:
        - images (list): list containing touples of (image [Tensor], image_name [string])
            example for loading images 'alan.jpg' : images[0] = (Tensor(1, 3, 256, 256), 'alan') where 4 is the num of channels
    '''
    
    images = []
    image_names = []

    for image_name in os.listdir(images_dir):

        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)
        image = torch.Tensor(image).float()
        image = torch.permute(image, (2, 0, 1))

        # Normalization:
        if normalize:
            image = image / 255
        
        image_id = image_name[:-4] # removing '.png' extension from image
        images.append((image, image_id))
        image_names.append(image_id)
        # images[image_id] = image  

    return images, image_names


def load_images_and_labels(images_dir_path, labels_path, classes, height_and_width_info):
    '''
    Loads images and labels. 

    Input:
        - images_path (string): path to directory which contains images
        - labels_path (string): path to a csv file which contains labels:
        - classes (list of strings): list of classes for example ["Alan Ford", "Sir Oliver", ...]
        - height_and_width_info (touple)
    
    Returns:
        - image_data [list of (Tensor, string) touples]: list which contains images
        - labels_data [list of Tensors]: list which contains labels
    '''
    
    images_data, image_names = load_images(images_dir_path)
    labels_data = load_labels(labels_path, image_names, classes, height_and_width_info)

    return images_data, labels_data


def augment_training_set(images, labels):
    '''
    Augments training data by aplying random small distortions on images.
    Images are slightly changed while labels remain unchanged for given image.
    Augmented images and labels are extended to images and labels lists in-place

    Inputs:
        - image_data [list of (Tensor, string) touples]: list which contains images
        - labels_data [list of Tensors]: list which contains labels

    '''
    augmented_images = []
    augmented_labels = []

    num = 0
    for (image, image_name), label in zip(images, labels):
        #print(f'augmenting image {num} from {len(images)}'); num += 1
        augmented_image = augment_image(image)
        augmented_images.append((augmented_image, "aug1_"+image_name))
        augmented_labels.append(label)
        augmented_image = augment_image(image)
        augmented_images.append((augmented_image, "aug2_"+image_name))
        augmented_labels.append(label)
        flipped_image, flipped_label = flip_image_and_annotation(image, label)
        augmented_images.append((flipped_image, "flip_"+image_name))
        augmented_labels.append(flipped_label)

    images.extend(augmented_images)
    labels.extend(augmented_labels)


def train_test_split(images, labels, test_percentage):
    '''
    Randomly splits images and labels into train and test dataset based on test_percentage.

    Input:
        - image_data [list of (Tensor, string) touples]: list which contains images
        - labels_data [list of Tensors]: list which contains labels
        - test_percentage (int): percentage of examples that will go to test dataset (others will go to training)
    
    Returns:
        - train_images [list of (Tensor, string) touples]: list which contains images for training 
        - train_labels [list of Tensors]: list which contains labels for training 
        - test_images [list of (Tensor, string) touples]: list which contains images for testing
        - test_labels [list of Tensors]: list which contains labels for testing
    '''

    total_length = len(images)
    test_data_len = int(total_length * test_percentage / 100)

    images_and_labels_zipped = list(zip(images, labels))
    random.shuffle(images_and_labels_zipped)
    images, labels = zip(*images_and_labels_zipped)

    #images = torch.stack(images)
    #labels = torch.stack(labels)


    # Shuffleling data
    #random_permutation = torch.randperm(total_length)
    #images = images[random_permutation]
    #labels = labels[random_permutation]

    test_images = list(images[:test_data_len])
    test_labels = list(labels[:test_data_len])
    train_images = list(images[test_data_len:])
    train_labels = list(labels[test_data_len:])
    
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


class YoloProductionDataset(Dataset):

    def __init__(self, images):
        self.images = images
        

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img = self.images[idx]
        return img
    

def make_torch_datasets(training_images, training_labels, test_images, test_labels):
    '''
    Wraps datasets in torch.utils.data.Dataset class and returns training and testing dataset
    '''
    train_dataset = YoloDetectionDataset(training_images, training_labels)
    test_dataset = YoloDetectionDataset(test_images, test_labels)

    return train_dataset, test_dataset


def production_dataloader(images_dir_path, batch_size=8):

    images_data, image_names = load_images(images_dir_path)
    
    dataset = YoloProductionDataset(images_data)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    return loader


def make_torch_dataloaders(images_dir_path, labels_path, classes, height_and_width_info, batch_size=8, augment=False):
    '''
    This function loads all images and labels, splits them randomly into training and test set and wraps these datasets in DataLoader class.
    '''

    #image_height, image_width, cell_width, cell_height, anchors = height_and_width_info
    

    images, labels = load_images_and_labels(images_dir_path, labels_path, classes, height_and_width_info)
    training_images, training_labels, test_images, test_labels = train_test_split(images, labels, test_percentage=20)
    
    if augment:
        augment_training_set(training_images, training_labels)

    train_dataset, test_dataset = make_torch_datasets(training_images, training_labels, test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(1302))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=torch.Generator().manual_seed(1302))

    return train_loader, test_loader


def main():
    print('test')
    images_dir_path = 'cropped_images'
    labels_path = 'annotations/Anal-Ford-color-export.csv'

    images_dir_path = 'C:\\Users\\cvetk\\Desktop\\projekat\\47_superhikov_veliki_poduhvat'
    labels_path = 'C:\\Users\\cvetk\\Desktop\\projekat\\anotacije\\af-export.csv'
    anchor_height = 250
    anchor_width = 300
    dimension_reduction = 32
    classes = ['Broj 1', 'Alan Ford', 'Bob Rok', 'Sir Oliver', 'Grunf', 'Jeremija', 'Sef']

    images, labels = load_images_and_labels(images_dir_path, labels_path, classes, anchor_height, anchor_width, dimension_reduction)
    
    train_images, train_labels, test_images, test_labels = train_test_split(images, labels, test_percentage=10)

    for (image, image_name), label in zip(train_images, train_labels):
        display_images_with_bounding_boxes(image, label, classes, 32, 32, anchor_width, anchor_height)

    print(f'train_images.shape = {train_images.shape}')
    print(f'train_labels.shape = {train_labels.shape}')
    print(f'test_images.shape = {test_images.shape}')
    print(f'test_labels.shape = {test_labels.shape}')

if __name__ == "__main__":
    main()