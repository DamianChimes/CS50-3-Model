import os
import numpy as np
import pandas as pd
from PIL import Image

def crop_image(img: Image) -> Image:
    """
    Crop and resize image to a square
    The image is cropped at the centre using the shorter length.
    It's then resized to the proposed dimensions.

    Args:
        img (Image): a PIL.Image object to be resized

    Returns:
        image_final (Image): a PIL.Image object that's resized
    """
    width, height = img.size
    diff = abs(width-height)

    # initialise final image parameters
    left, right, top, bottom = 0, width, 0, height

    # crop based on whether the difference in dimensions is odd or even
    if diff % 2 == 0:
        if width >= height:
            bottom = height
            left = diff / 2
            right = width - left
        elif height > width:
            top = diff / 2
            bottom = height - top
            right = width
    else:
        if width > height:
            bottom = height
            left = diff / 2 + 0.5
            right = width - left + 1
        elif height > width:
            top = diff / 2 + 0.5
            bottom = height - top + 1
            right = width

    # crop image into a square
    img_cropped = img.crop((left, top, right, bottom))
    # resize to desired shape
    img_final = img_cropped.resize((224, 224))
    
    return img_final

def load_images(image_df: pd.DataFrame) -> np.array:
    """
    Loads in the images based on the input image metadata

    Arguments:
        image_df (pd.DataFrame): DF of images metadata, which includes their file paths

    Returns:
        images (np.array): A (no of images) x (img_size^2 x channels) array containing images specified in image_df
    """
    images = []

    for idx, file_path in enumerate(image_df['filepath']):
        img = Image.open(file_path)
        img = img.convert('RGB')
        img_final = crop_image(img)
        img_array = np.asarray(img_final).flatten()

        images.append(img_array)

    images = np.array(images)

    return images

def load_labels(image_df: pd.DataFrame) -> np.array:
    """
    Loads in the image labels based on the input image metadata

    Arguments:
        image_df (pd.DataFrame): DF of images metadata, which includes their file paths

    Returns:
        labels (np.array): A (no of images x 1) array containing image labels
    """
    labels = []

    label_encode = {
        'positive': 1,
        'negative': 0,
        'COVID-19': 2,
        'pneumonia': 1,
        'normal': 0
    }

    for label in image_df['class']:
        labels.append(label_encode.get(label))

    labels = np.array(labels)

    return labels

label = pd.read_csv('./data/external/evdata_V2.txt', header=None, sep=' ')

label.columns = ['patient_id', 'filename', 'class', 'data_source']
label['filepath'] = label.apply(lambda row: os.path.join('./data/external/evdata_V2', row['filename']), axis=1)

X_external = load_images(label)
Y_external = load_labels(label)

np.save('./data/input/xexternal.npy', X_external)
np.save('./data/input/yexternal.npy', Y_external)