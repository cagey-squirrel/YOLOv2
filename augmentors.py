'''
This script is used for augmenting images
Images on input are Tensors
'''

import numpy as np
from torchvision.transforms.functional import affine, adjust_contrast
import torch 
import random
from matplotlib import pyplot as plt


def augment_image(image):
    '''
    This function does random augmentation of an image
    Input:
        -image (np Tensor): single image
    Output:
        -augmented image (Tensor)
    '''
    chance = random.uniform(0, 1)

    # random augmentation
    if chance < 0.25:
        return rotate_image(image)
    if chance < 0.5:
        return scale_image(image)                                                                                                 
    if chance < 0.75:
        return shear_image(image)
    else:
        return change_image_contrast(image)


def flip_image_and_annotation(image, label):
    '''
    This function flips image and its label along y-axis

    Input:
        -image (np.array): single image
        - anotation (Tensor): single label
    Output:
        -flipped_image (np.array)
        -flipped_label (Tensor)
    '''
    image = image.permute((1, 2, 0))
    image = image.numpy()
    flipped_image = np.fliplr(image)
    flipped_image = flipped_image.transpose((2, 0, 1))
    flipped_image = torch.from_numpy(flipped_image.copy())
    
    flipped_label = torch.zeros(label.shape)
    non_zero_indices = (label[..., 4] > 0).nonzero(as_tuple=True)
    
    if non_zero_indices[0].nelement() == 0:
        return flipped_image, flipped_label

    coords1, coords2, coords3 = non_zero_indices

    for coord1, coord2, coord3 in zip(coords1, coords2, coords3):
        original_label = label[coord1, coord2, coord3].detach().clone()
        original_label[0] = 18 - original_label[0]
        flipped_label[18-coord1, coord2, coord3] = original_label

    return flipped_image, flipped_label



def rotate_image(image, angle=5):
    '''
    Rotates images for angle degrees in clock-wise direction
    Inputs:
        - image (np array)
        - angle (int): number of degrees that images will be rotated
    Returns:
        - rotated_image (np array): rotated image
    '''

    # affine takes the tensor of shape (C, H, W) so we need to add a dummy channel dim
    image = image[None, :, :]
    
    rotated_image = affine(image, angle=angle, translate=[0, 0], scale=1, shear=0)

    # Now we drop the dummy dimension C:
    rotated_image = rotated_image.squeeze()
    
    return rotated_image


def scale_image(image, scale=1.1):
    '''
    Zooms image scale times
    Inputs:
        - image (Tensor)
        - scale (int): scale decalaring how much to zoom in a pic
    Returns:
        - scaled_image (Tensor)
    '''

    # affine takes the tensor of shape (C, H, W) so we need to add a dummy channel dim
    image = image[None, :, :]
    
    scaled_image = affine(image, angle=0, translate=[0, 0], scale=scale, shear=0)
    
    # Now we drop the dummy dimension C:
    scaled_image = scaled_image.squeeze()

    return scaled_image


def shear_image(image, shear=5):
    '''
    Zooms images scale times
    Inputs:
        - image (Tensor)
        - shear (int): decalring how much to shear an image
    Returns:
        - sheard_image (Tensor)
    '''

    # affine takes the tensor of shape (C, H, W) so we need to add a dummy channel dim
    image = image[None, :, :]
    
    sheard_image = affine(image, angle=0, translate=[0, 0], scale=1, shear=shear)

    # Now we drop the dummy dimension C:
    sheard_image = sheard_image.squeeze()

    return sheard_image


def change_image_contrast(image, contrast=0.9):
    '''
    Changing contrast of images
    Inputs:
        - image (Tensor)
        - contrast (int): contrast which will be applied to picture
    Returns:
        - contrasted_image (Tensor)
    '''

    # affine takes the tensor of shape (C, H, W) so we need to add a dummy channel dim
    image = image[None, :, :]
    
    contrasted_image = adjust_contrast(image, contrast)

    # Now we drop the dummy dimension C:
    contrasted_image = contrasted_image.squeeze()

    return contrasted_image