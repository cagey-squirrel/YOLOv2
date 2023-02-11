import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from skimage.transform import resize
import matplotlib.image
from collections import defaultdict
#im = Image.fromarray(A)
#im.save("your_file.jpeg")

def resize_images(images_dir, output_dir, output_shape):
    for img_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, img_name)
        image = np.array(Image.open(f"{image_path}"))
        print(image.shape)
        image = resize(image, output_shape=output_shape)
        img_output_path = os.path.join(output_dir, img_name)
        matplotlib.image.imsave(img_output_path, image)

def print_images_shape(images_dir):
    for img_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, img_name)
        image = np.array(Image.open(f"{image_path}"))
        print(image.shape)
        #plt.imshow(image)
        #plt.show()


def crop_resize_and_save_images(src_dir, dst_dir, output_image_shape, limit=float('inf')):
    '''
    Each image from src dir contains 2 images: one at top and one at bottom.
    This function splits this image into two separate images, resizes them and saves them at dst_dst dir
    '''

    # All images start from 20px from the left and end at 880 pixels at the right
    left_edge = 20
    right_edge = 880
    images_done = 0

    for image_name in list(os.listdir(src_dir))[3:-1]: # skipping first 3 and last page (usually not usefull for detection)
        image_path = os.path.join(src_dir, image_name)
        image = np.array(Image.open(f"{image_path}"))

        # Top image starts at 20pixels at ends at about half of image (49%)
        top_edge = 20
        bottom_edge = int(image.shape[0] * 0.49)
        top_image = image[top_edge:bottom_edge, left_edge:right_edge]

        # First image starts at 5px after top image, and ends at about 97% of image
        top_edge = bottom_edge + 5
        bottom_edge = int(image.shape[0] * 0.97)
        bot_image = image[top_edge:bottom_edge, left_edge:right_edge]

        top_image = resize(top_image, output_shape=output_image_shape)
        bot_image = resize(bot_image, output_shape=output_image_shape)

        dst_top_image_path = os.path.join(dst_dir, image_name[:-4] + "_high.jpg")
        dst_bot_image_path = os.path.join(dst_dir, image_name[:-4] + "_low.jpg")

        matplotlib.image.imsave(dst_top_image_path, top_image)
        matplotlib.image.imsave(dst_bot_image_path, bot_image)

        images_done += 1
        if images_done > limit:
            break

def rename_images(src_dir):

    for image_name in list(os.listdir(src_dir)):
        old_image_path = os.path.join(src_dir, image_name)

        image_name_parts = image_name.split('_')
        num = str(image_name_parts[-1])
        num = '0' * (7 - len(num)) + num
        image_name_parts[-1] = num

        new_image_name = '_'.join(image_name_parts)
        new_image_path = os.path.join(src_dir, new_image_name)

        os.rename(old_image_path, new_image_path)

        

def main():

    dir_names = [
        '14_udar_munje',
        '25_superhik',
        '31_bi_bim_ba_bam',
        '34_dvanaest_umetnika',
        '35_centurion',
        '44_derbi',
        '45_tako_je_nast_TNT',
        '46_povratak_superhika',
        '47_superhikov_veliki_poduhvat'    
    ]

    for dir_name in dir_names:
        crop_resize_and_save_images(dir_name, "cropped_images", output_image_shape=(388, 576))
        #rename_images(dir_name)


if __name__ == "__main__":
    main()