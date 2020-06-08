import matplotlib.pyplot as plt


def plot_imgs(pred_Im0, pred_Im1, noisy0, noisy1):
    slice = 10
    fig, ax = plt.subplots(3, 2, figsize=(20, 30))
    plt.subplots_adjust(hspace=0, wspace=0)

    ax[0, 0].set_title('noisy0', fontsize=30)
    ax[0, 0].imshow(noisy0[:, :, slice], cmap=plt.get_cmap('gray'))

    ax[0, 1].set_title('pred_Im0', fontsize=30)
    ax[0, 1].imshow(pred_Im0[:, :, slice], cmap=plt.get_cmap('gray'))

    ax[1, 0].set_title('noisy1', fontsize=30)
    ax[1, 0].imshow(noisy1[:, :, slice], cmap=plt.get_cmap('gray'))

    ax[1, 1].set_title('pred_Im1', fontsize=30)
    ax[1, 1].imshow(pred_Im1[:, :, slice], cmap=plt.get_cmap('gray'))

    ax[2, 0].set_title('mean noisy', fontsize=30)
    ax[2, 0].imshow((noisy1[:, :, slice] + noisy0[:, :, slice])/2, cmap=plt.get_cmap('gray'))

    ax[2, 1].set_title('mean pred', fontsize=30)
    ax[2, 1].imshow((pred_Im1[:, :, slice] + pred_Im0[:, :, slice])/2, cmap=plt.get_cmap('gray'))

    plt.xticks([]), plt.yticks([])
    plt.show()
