from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, Activation
from keras import initializers

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm = True, seed = 0):
  initializer = initializers.he_normal(seed = seed)  
    
  # First layer
  x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
             kernel_initializer = initializer, padding = 'same')(input_tensor)
  
  if batchnorm: 
    x = BatchNormalization()(x)
  
  x = Activation('relu')(x)

  # Second layer
  x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),
             kernel_initializer = initializer, padding = 'same')(input_tensor)
  
  if batchnorm: 
    x = BatchNormalization()(x)
  
  x = Activation('relu')(x)

  return x
