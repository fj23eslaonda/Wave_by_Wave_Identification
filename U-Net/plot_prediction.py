import matplotlib.pyplot as plt

def plot_prediction (X, y, preds, binary_preds, index):

    fig, ax = plt.subplots(2, 2, figsize = (10,10))

    # Figure 1
    ax[0,0].imshow(X[index, ..., 0], cmap = 'gray')
    ax[0,0].grid(False)
    ax[0,0].contour(y[index].squeeze(), color = 'r', level = [0.5])
    ax[0,0].set_title('Video image + Original mask')

    # Figure 2
    ax[0,1].imshow(y[index].squeeze(), cmap = 'gray')
    ax[0,1].grid(False)
    ax[0,1].set_title('Orginal mask')

    # Figure 3
    ax[1,0].imshow(preds[index].squeeze(), cmap = 'gray')
    ax[1,0].grid(False)
    ax[1,0].set_title('Prediction mask')

    # Figure 4
    ax[1,1].imshow(binary_preds[index].squeeze(), cmap = 'gray')
    ax[1,1].grid(False)
    ax[1,1].set_title('Prediction binary mask')

    
