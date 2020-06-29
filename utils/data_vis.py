from matplotlib import pyplot as plt
from IPython.display import clear_output


def plot_imgs(rec_imgs, F_rec_Kspaces, fully_sampled_img, ZF_img):
    plt.figure()

    slices = [20, 40, 50, 60, 70, 90]

    # for slice in range(rec_imgs.shape[2])
    for slice in slices:
        fig, ax = plt.subplots(1, 4, figsize=(40, 10))
        plt.subplots_adjust(hspace=0, wspace=0)
        ax[0].set_title('Final reconstruction', fontsize=30)
        ax[0].imshow(rec_imgs[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        ax[1].set_title('Kspace reconstruction', fontsize=30)
        ax[1].imshow(F_rec_Kspaces[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))


        ax[2].set_title('ZF', fontsize=30)
        ax[2].imshow(ZF_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))

        ax[3].set_title('Fully sampled image', fontsize=30)
        ax[3].imshow(fully_sampled_img[:, :, slice], vmin=0, vmax=1, cmap=plt.get_cmap('gray'))

        plt.xticks([]), plt.yticks([])
        plt.show()
        clear_output(wait=True)


