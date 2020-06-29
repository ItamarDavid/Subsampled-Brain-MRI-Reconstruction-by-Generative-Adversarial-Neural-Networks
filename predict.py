import argparse
import logging
import os

import numpy as np
import torch

import glob
import h5py
import pickle

from Networks.generator_model import WNet
from utils.data_vis import plot_imgs
from utils.data_save import save_data


def crop_toshape(kspace_cplx, args):
    if kspace_cplx.shape[0] == args.img_size:
        return kspace_cplx
    if kspace_cplx.shape[0] % 2 == 1:
        kspace_cplx = kspace_cplx[:-1, :-1]
    crop = int((kspace_cplx.shape[0] - args.img_size) / 2)
    kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]

    return kspace_cplx


def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2(kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]


def slice_preprocess(kspace_cplx, slice_num, masks, maskedNot, args):
    # crop to fix size
    kspace_cplx = crop_toshape(kspace_cplx, args)
    # split to real and imaginary channels
    kspace = np.zeros((args.img_size, args.img_size, 2))
    kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
    kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
    # target image:
    image = ifft2(kspace_cplx)

    # HWC to CHW
    kspace = kspace.transpose((2, 0, 1))
    masked_Kspace = kspace * masks[:, :, slice_num]
    masked_Kspace += np.random.uniform(low=args.minmax_noise_val[0], high=args.minmax_noise_val[1],
                                       size=masked_Kspace.shape) * maskedNot

    return masked_Kspace, kspace, image


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')

    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    args = parser.parse_args()
    args.bilinear = True
    args.mask_path = './Masks/mask_20_256.pickle'
    args.img_size = 256
    args.viz = True
    args.save = False
    args.save_path = ''


    args.NumInputSlices = 3
    args.minmax_noise_val = [-0.01, 0.01]
    args.masked_kspace = True
    args.model = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_multi/CP_epoch57.pth'

    args.input_path = '/HOME/reconstructed/data/IXIhdf5/test/'
    return args


if __name__ == "__main__":
    args = get_args()
    gpu_id = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logging.info(f'Using device {device}')

    gen_net = WNet(args, masked_kspace=args.masked_kspace)

    logging.info("Loading model {}".format(args.model))


    gen_net.to(device=device)

    checkpoint = torch.load(args.model, map_location=device)
    gen_net.load_state_dict(checkpoint['model_state_dict'])

    gen_net.eval()

    logging.info("Model loaded !")

    in_files = glob.glob(args.input_path + '*.hdf5')

    mask_path = args.mask_path
    with open(mask_path, 'rb') as pickle_file:
        masks_dictionary = pickle.load(pickle_file)
    masks = np.dstack((masks_dictionary['mask0'], masks_dictionary['mask1'], masks_dictionary['mask2']))
    maskNot = 1 - masks_dictionary['mask1']

    for i, infile in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(infile))

        with h5py.File(infile, 'r') as f:
            img_shape = f['data'].shape
            fully_sampled_img = f['data'][:]

        #preprocess data:
        rec_imgs = np.zeros(img_shape)
        rec_Kspaces = np.zeros(img_shape, dtype=np.csingle) #comples
        F_rec_Kspaces = np.zeros(img_shape)
        ZF_img = np.zeros(img_shape)

        for slice_num in range(img_shape[2]):
            add = int(args.NumInputSlices / 2)
            with h5py.File(infile, 'r') as f:
                if slice_num == 0:
                    imgs = np.dstack((f['data'][:, :, 0], f['data'][:, :, 0], f['data'][:, :, 1]))
                elif slice_num == img_shape[2]-1:
                    imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num]))
                else:
                    imgs = np.dstack((f['data'][:, :, slice_num-1], f['data'][:, :, slice_num], f['data'][:, :, slice_num + 1]))

            masked_Kspaces_np = np.zeros((args.NumInputSlices * 2, args.img_size, args.img_size))
            target_Kspace = np.zeros((2, args.img_size, args.img_size))
            target_img = np.zeros((1, args.img_size, args.img_size))

            for sliceNum in range(args.NumInputSlices):
                img = imgs[:, :, sliceNum]
                kspace = fft2(img)
                slice_masked_Kspace, slice_full_Kspace, slice_full_img = slice_preprocess(kspace, sliceNum,
                                                                                          masks, maskNot, args)
                masked_Kspaces_np[sliceNum * 2:sliceNum * 2 + 2, :, :] = slice_masked_Kspace
                if sliceNum == int(args.NumInputSlices / 2):
                    target_Kspace = slice_full_Kspace
                    target_img = slice_full_img

            masked_Kspaces = np.expand_dims(masked_Kspaces_np, axis=0)

            masked_Kspaces = torch.from_numpy(masked_Kspaces).to(device=args.device, dtype=torch.float32)

            #predict:
            rec_img, rec_Kspace, F_rec_Kspace = gen_net(masked_Kspaces)

            rec_img = np.squeeze(rec_img.data.cpu().numpy())
            rec_Kspace = np.squeeze(rec_Kspace.data.cpu().numpy())
            rec_Kspace = (rec_Kspace[0, :, :] + 1j*rec_Kspace[1, :, :])

            F_rec_Kspace = np.squeeze(F_rec_Kspace.data.cpu().numpy())

            rec_imgs[:, :, slice_num] = rec_img
            rec_Kspaces[:, :, slice_num] = rec_Kspace
            F_rec_Kspaces[:, :, slice_num] = F_rec_Kspace
            ZF_img[:, :, slice_num] = np.squeeze(ifft2((masked_Kspaces_np[2, :, :] + 1j*masked_Kspaces_np[3, :, :])))



        if args.save:
            os.makedirs(args.save_path, exist_ok=True)
            out_file_name = args.save_path + os.path.split(infile)[1]
            save_data(out_file_name. rec_imgs, F_rec_Kspaces, fully_sampled_img, ZF_img, rec_Kspaces)

            logging.info("reconstructions save to: {}".format(out_file_name))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(infile))
            plot_imgs(rec_imgs, F_rec_Kspaces, fully_sampled_img, ZF_img)
