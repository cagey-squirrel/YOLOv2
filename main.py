import numpy as np
import random
from time import time
import os
from train import training

def main():
    np.random.seed(1302)
    random.seed(1302)
    images_dir_path = 'C:\\Users\\cvetk\\OneDrive\\Desktop\\Master\\KV\\cropped_images_small'
    labels_path = 'annotations//Alan-Ford-color-export.csv'
    labels_path = 'annotations//annotations_small.csv'
    classes = ['Broj 1', 'Alan Ford', 'Bob Rok', 'Sir Oliver', 'Grunf', 'Jeremija', 'Sef']

    output_dir_name = 'first_training2' + str(time())
    output_dir_path = os.path.join('output', output_dir_name)
    os.mkdir(output_dir_path)

    image_height = 384
    image_width = 576
    num_cells_height = 2
    num_cells_width = 8

    anchors = \
        [
            (150, 150),
            (250, 250)
        ]
    anchors = np.array(anchors)
    cell_width = 72
    cell_height = 192

    
    height_and_width_info = image_height, image_width, cell_width, cell_height, anchors
    
    training(images_dir_path, labels_path, classes, 1500, height_and_width_info, output_dir_path)


if __name__ == "__main__":
    main()