# Transform each image in row vector 
# and return a matrix with all image of each set

import numpy as np
from tqdm import tqdm_notebook
from keras.preprocessing.image import array_to_img, img_to_array, load_img

def create_inputs(ids,path_image, path_label, im_height, im_width):
  # Create matrix
  X = np.zeros((len(ids), im_height, im_width, 1), dtype = np.float32)
  Y = np.zeros((len(ids), im_height, im_width,1), dtype = np.float32)
  ids.sort()

  for i, index in tqdm_notebook(enumerate(ids), total = len(ids)):

    # load image and transform image to array
    x_img   = img_to_array(load_img(path_image+index, grayscale = True))

    # load label and tranform image to array
    mask = img_to_array(load_img(path_label+index, grayscale = True))

    # Save images
    X[i] = x_img /255.0
    Y[i] = mask  / 255.0
  
  return X, Y
