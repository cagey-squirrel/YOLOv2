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

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

def get_labels_from_annotations(annotations, image_height, image_width):
    num_cells_height = 12 
    numm_cells_width = 18
    anchor_height = 250
    anchor_width = 300
    cell_width = 32
    cell_height = 32

    labels = defaultdict(lambda: [])

    # Getting indexes of labels
    classes = ['Alan', 'Jeremija', 'Brok', 'Broj 1', 'Sir Oliver', 'Grunt']
    number_of_classes = len(classes)
    agent_name_to_class_number = {}
    for i, agent_name in enumerate(classes):
        agent_name_to_class_number[agent_name] = i
    
    # Prva 4 elementa labele su koordinate i velicina
    # label = [box_center_x, box_center_y, box_width, box_height, object_is_in_this_cell, one_hot_class_encodings]
    for image_name in annotations:

        image_labels = np.zeros(shape=(1, numm_cells_width, num_cells_height, (5 + number_of_classes)))

        annotation_list = annotations[image_name]
        label_list = []
        for annotation in annotation_list:
            box_center_x, box_center_y, box_width, box_height, class_name = get_center_and_size_of_annotation(annotation, image_height, image_width, cell_height, cell_width, anchor_height, anchor_width)
            
            box_center_cell_index_x = floor(box_center_x)
            box_center_cell_index_y = floor(box_center_y)

            label = [box_center_x, box_center_y, box_width, box_height]
            label.append(1) # object is in this cell -> certainty is 1
            class_one_hot_encoding = [0 for _ in range(number_of_classes)]
            class_one_hot_encoding[agent_name_to_class_number[class_name]] = 1
            label.extend(class_one_hot_encoding)
            
            #label_list.append(label)
            label = np.array(label)
            image_labels[0, box_center_cell_index_x, box_center_cell_index_y, :] = label

        #label_list = np.array(label_list)
        #label_list = torch.Tensor(label_list)
        label_list = torch.Tensor(image_labels)
        labels[image_name] = label_list
    
    #labels = np.array(labels, dtype=np.float16)
    #labels = torch.Tensor(labels)
    return labels

    # Za svako polje labela oznacavamo da nema ni jednog podatka
    # Za ona polja gde ima objekta stavljamo sve vrednosti podatka 
    # Ono sto treba da vratimo ima dimenzije 3x3x9 tj br_celija_po_sirini x br_celija_po_duzini x (5+br_klasa)
    # U trenutku odredjivanja labele treba nam da znamo koliko celija ce biti 
    # Sa puno celija ne trebaju nam anchor boxevi

def get_center_and_size_of_annotation(annotation, image_height, image_width, cell_height, cell_width, anchor_height, anchor_width):
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



def load_annotations(annotations_path):
    
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
    

def display_images_with_bounding_boxes(image, bounding_boxes, cell_width, cell_height, anchor_width, anchor_height, name=''):
    '''
    Displays images with bounding boxes around them

    Input:
        - image (Tensor of shape: ): image for display
        - bounding_boxes (Tensor containing bounding box tensors): bounding boxes which will be displayed
            each bounding box is tensor of shape: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
            CLASS_ONE_HOT_ENCODING contains one element for each class 
    Returns:
        - None
    '''

    image = torch.permute(image, (1, 2, 0)).int()
    fig, axis = plt.subplots()
    axis.imshow(image)
    image = image.detach().numpy()
    bounding_boxes = bounding_boxes.detach().numpy()

    for bounding_box in bounding_boxes:

        if not bounding_box.any():
            continue
        
        # Bounding box contains: (box_center_x, box_center_y, box_width, box_height, confidence, CLASS_ONE_HOT_ENCODING)
        # Box_center_x, and Box_center_y are in units of cells so we need to multiply them to display them on image
        # box_width and box_height are in units of anchor sizes so they too need to be multiplied in order to be displayed
        #print(f'bbox = {bounding_box}')
        box_center_x = bounding_box[0] * cell_width
        box_center_y = bounding_box[1] * cell_height 

        box_width = bounding_box[2] * anchor_width 
        box_height = bounding_box[3] * anchor_height

        # Class Rectangle needs top left corner as input
        top_left_corner_x = box_center_x - box_width / 2
        top_left_corner_y = box_center_y - box_height / 2

        edge_color = 'red'

        rect = matplotlib.patches.Rectangle((top_left_corner_x, top_left_corner_y), box_width, box_height, fill=False, edgecolor=edge_color) 
        axis.add_patch(rect)
    
    plt.show()



def show_images_with_boxes(input_tensor, output_tensor, cell_height, cell_width, anchor_height, anchor_width):

    to_img = torchvision.transforms.ToPILImage()
    for img, predictions in zip(input_tensor, output_tensor):
        # img = to_img(img)
        if 0 in predictions.shape: # empty tensor
            # display(img)
            print('empty')
            continue
        print(f'predictions = {predictions}')
        print(f'predictions.shape = {predictions.shape}')
        # ... dodje do nizova koji sadrze elemente
        # iz svakog elementarnog niza uzmi cetvrti element (sigurnost da postoji objekat)
        confidences = predictions[..., 4].flatten()
        
        # Od svih elementarnih nizova uzmi prva 4 elementa (koord i velicinu)
        # contiguous radi poravnanje u memoriji
        # view(-1, 4) -> -1 radi flatten a 4 ga dodatno deli na nizove od po 4 elementa
        # Dobija se matrica br_predvidjanja x 4
        boxes = (
            predictions[..., :4].contiguous().view(-1, 4)
        )

        # Uzima samo verovatnoce klasa iz predikcija
        # Daje matricu br_predvidjanja x br_klasa
        classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)

        # Prvi i treci element pomnozi sirinom slike (x i sirina * width)
        boxes[:, 0] *= cell_height
        boxes[:, 1] *= cell_width
        boxes[:, 2] *= anchor_height
        boxes[:, 3] *= anchor_width
        #boxes[:, ::2] *= img.width
        

        # Drugi i cetvrti element pomnozi visinom slike (y i visina * height)
        #boxes[:, 1::2] *= img.height

        # Gornji levi ugao kvadrata je ulevo i gore za pola kocke
        # A donji desni ugao je dole i udesno za pola kocke
        boxes = (torch.stack([
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
        ], -1, ).cpu().to(torch.int32).numpy())

        f,ax = plt.subplots()
        img = torch.permute(img, (1, 2, 0))
        print(img.shape)
        ax.imshow(img[..., 0])
        ind = 0
        for box, confidence, class_ in zip(boxes, confidences, classes):
            if confidence < 0.01:
                print("low conf")
                continue # don't show boxes with very low confidence
            # make sure the box fits within the picture:
            
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img.shape[0] - 1, int(box[2])),
                min(img.shape[1] - 1, int(box[3])),
            ]
            
            try:  # either the class is given as the sixth feature
                idx = int(class_.item())
            # ovo
            except ValueError:  # or the 20 softmax probabilities are given as features 6-25
                print('except')
                idx = int(torch.max(class_, 0)[1].item())
            try:
                class_ = CLASSES[idx]  # the first index of torch.max is the argmax.
            except IndexError: # if the class index does not exist, don't draw anything:
                print("no class")
                continue

            
            color = (  # green color when confident, red color when not confident.
                int((1 - (confidence.item())**0.8 ) * 255),
                int((confidence.item())**0.8 * 255),
                0,
            )
            #draw = ImageDraw.Draw(img)
            #draw.rectangle(box, outline=color)
            #draw.text(box[:2], class_, fill=color)

            if confidence.item() > 0.8:
                edge_color = 'green'
            else:
                edge_color = 'red'

            edge_color = 'red' if ind > 8 else 'green'
            edge_color = 'red'
            
            rect = matplotlib.patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=edge_color)
            ax.add_patch(rect)
            plt.show()
            ind += 1
        print(f'ind = {ind}')
        #display(img)
        
        #plt.show()

def load_images(images_dir):
    '''
    Loads images from image_dir and returnes them as tensors

    Input:
        - images_dir (string): path to directory which contains images
    
    Returns:
        - images (directory): directory containing images names as keys and tensors as images
            example for loading images 'alan.jpg' : images['alan'] = Tensor(1, 4, 256, 256) where 4 is the num of channels
    '''
    
    images = {}
    for image_name in os.listdir(images_dir):

        image_path = os.path.join(images_dir, image_name)
        image = np.array(Image.open(image_path))
        image = torch.Tensor(image)
        image = torch.permute(image, (2, 0, 1))
        #image = image[None, :, :, :]
        
        image_id = image_name[:-4] # removing '.png' extension from image
        images[image_id] = image  

    return images

def main():
    
    annotations = load_annotations('annotations.csv')
    #basic_plot_annotations(annotations)
    
    for image_name in annotations.keys():
        
        image = np.array(Image.open(f"{image_name}.png"))
        image = torch.Tensor(image)
        image = torch.permute(image, (2, 0, 1))
        image = image[None, :, :, :]
        
        print(f'image shape in draw = {image.shape}')
        annotation_list = annotations[image_name]
        for annotation in annotation_list:
            center_x, center_y, width, height, label = get_center_and_size_of_annotation(annotation, image.shape[2], image.shape[3])
            print(f'center_x, center_y, width, height = {center_x}, {center_y}, {width}, {height}')
            label = [center_x, center_y, width, height, 1, 0, 0, 1, 0]
            label = torch.Tensor(np.array(label))
            label = label[None, None, None, None, :]

            show_images_with_boxes(image, label)
            print(f'label shape in draw = {label.shape}')

if __name__ == '__main__':
    main()
        