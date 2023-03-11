import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from collections import defaultdict
from nltk.cluster.kmeans import KMeansClusterer


def get_height_and_width_of_annotation(annotation):
    '''
    Calculates height and width of bounding box
    '''
    top_left_corner_y, top_left_corner_x, bot_right_corner_y, bot_right_corner_x  = annotation
    box_height = bot_right_corner_y - top_left_corner_y
    box_width = bot_right_corner_x - top_left_corner_x
    
    return box_width, box_height


def get_bounding_boxes_sizes(annotations_path):
    '''
    Returns widths and heights from bounding boxes in train set
    '''

    annotations = load_annotations(annotations_path)

    sizes = []


    for image_name in annotations:
        annotation_list = annotations[image_name]
        for annotation in annotation_list:
            box_width, box_height = get_height_and_width_of_annotation(annotation)
            sizes.append([box_width, box_height])

    return np.array(sizes)


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

        annotations[image_name].append((top_left_corner_y, top_left_corner_x, bot_right_corner_y, bot_right_corner_x))
    
    return annotations


def find_best_anchor_sizes(annotations_path):

    sizes = get_bounding_boxes_sizes(annotations_path)
    clst = KMeansClusterer(2, distance=metric, repeats=25)
    clst.cluster(sizes, assign_clusters=True)
    
    print(clst.means())


def metric(box1, box2):

    min_width = min(box1[0], box2[0])
    min_height = min(box1[1], box2[1])

    intersection = min_width * min_height
    area1 = box1[0] * box1[1]
    area2 = box2[0] * box2[1]

    intersection_over_union = intersection / (area1 + area2 - intersection)

    # This is distance metric so it needs to be high for different bounding boxes
    # and low for similar ones
    return 1 - intersection_over_union

def main():
    annotations_path = 'annotations//Alan-Ford-color-export.csv'
    annotations_path = 'annotations//Alan_samo_lica.csv'
    find_best_anchor_sizes(annotations_path)


if __name__ == '__main__':
    main()
    