import numpy as np
import random
from time import time
import os
from train import training
from test import test
import json

def main():
    np.random.seed(1302)
    random.seed(1302)

    input_params_file = open('input_params.json')
    input_params = json.load(input_params_file)

    image_height = 384
    image_width = 576
    num_cells_height = 2
    num_cells_width = 8

    anchors = \
        [
            #(175, 175),
            (150, 150),
            (250, 250)
        ]
    anchors = np.array(anchors)
    cell_width = 72
    cell_height = 192

    
    height_and_width_info = image_height, image_width, cell_width, cell_height, anchors
    classes = ['Broj 1', 'Alan Ford', 'Bob Rok', 'Sir Oliver', 'Grunf', 'Jeremija', 'Sef']
    
    training(classes, height_and_width_info, input_params)
    #test(classes, height_and_width_info, input_params)


if __name__ == "__main__":
    main()