import matplotlib.pyplot as plt

def plot_prediction (X, y, binary_preds, index):

    fig, ax = plt.subplots(2, 3, figsize = (15,10))

    # Figure 1
    ax[0,0].imshow(X[index, ..., 0], cmap = 'gray')
    ax[0,0].grid(False)
    ax[0,0].set_title('Video image')

    # Figure 2
    ax[0,1].imshow(y[index].squeeze(), cmap = 'gray')
    ax[0,1].grid(False)
    ax[0,1].set_title('Orginal mask')
    
    # Figure 3
    ax[0,2].imshow(X[index, ..., 0], cmap = 'gray')
    ax[0,2].grid(False)
    ax[0,2].contour(y[index].squeeze(), colors = 'r', levels = [0.5])
    ax[0,2].set_title('Video image + Original mask')

     # Figure 2
    ax[1,0].imshow(X[index, ..., 0], cmap = 'gray')
    ax[1,0].grid(False)
    ax[1,0].set_title('Video image')

    # Figure 4
    ax[1,1].imshow(binary_preds[index].squeeze(), cmap = 'gray')
    ax[1,1].grid(False)
    ax[1,1].set_title('Prediction binary mask')
    
    # Figure 3
    ax[1,2].imshow(X[index, ..., 0], cmap = 'gray')
    ax[1,2].grid(False)
    ax[1,2].contour(binary_preds[index].squeeze(), colors = 'r', levels = [0.5])
    ax[1,2].set_title('Video image + Predicted mask')
    
    plt.show()

    
