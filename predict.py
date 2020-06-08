import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import glob
import h5py

from Generator import WNet
from utils.data_vis import plot_imgs
from utils.data_save import save_data
from utils.dataset import BasicDataset, AspectDataset_multi


def crop_toshape(kspace_cplx, args):
    if kspace_cplx.shape[0] == args.img_size:
        return kspace_cplx
    if kspace_cplx.shape[0] % 2 == 1:
        kspace_cplx = kspace_cplx[:-1, :-1]
    crop = int((kspace_cplx.shape[0] - args.img_size) / 2)
    kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]

    return kspace_cplx


def ifft2(kspace_cplx):
    return np.absolute(np.fft.fftshift(np.fft.ifft2(kspace_cplx)))[None, :, :]


def preprocess(kspace_cplx, args):
    kspace_cplx = crop_toshape(kspace_cplx, args)

    kspace_in = np.zeros((2, args.img_size, args.img_size))
    kspace_in[0, :, :] = np.real(kspace_cplx).astype(np.float32)
    kspace_in[1, :, :] = np.imag(kspace_cplx).astype(np.float32)


    img= ifft2(kspace_cplx)


    return kspace_in, img

def predict(net, input0, input1, device, args):
    net.eval()
    net_input0 = np.zeros((2*args.NumInputSlices, args.img_size, args.img_size))
    net_input1 = np.zeros((2*args.NumInputSlices, args.img_size, args.img_size))

    noisy_img0 = np.zeros((args.img_size, args.img_size))
    noisy_img1 = np.zeros((args.img_size, args.img_size))
    for slice in range(args.NumInputSlices):
        kspace_0_ = input0[:, :, slice]
        kspace_1_ = input1[:, :, slice]

        kspace_0_, img_0 = preprocess(kspace_0_, args)
        kspace_1_, img_1 = preprocess(kspace_1_, args)


        net_input0[slice * 2:slice * 2 + 2, :, :] = kspace_0_
        net_input1[slice * 2:slice * 2 + 2, :, :] = kspace_1_

        if slice ==1:
            noisy_img0 = img_0
            noisy_img1 = img_1


    net_input0 = torch.from_numpy(net_input0).unsqueeze(0)
    net_input1 = torch.from_numpy(net_input1).unsqueeze(0)


    net_input0 = net_input0.to(device=device, dtype=torch.float32)
    net_input1 = net_input1.to(device=device, dtype=torch.float32)


    with torch.no_grad():

        pred_img0, pred_kspace0, F_pred_kspace0 = net(net_input0)
        pred_img1, pred_kspace1, F_pred_kspace1 = net(net_input1)

    pred_img0 = pred_img0.data.cpu().numpy()
    F_pred_kspace0 = F_pred_kspace0.data.cpu().numpy()

    pred_img1 = pred_img1.data.cpu().numpy()
    F_pred_kspace1 = F_pred_kspace1.data.cpu().numpy()




    return pred_img0, F_pred_kspace0, noisy_img0, pred_img1, F_pred_kspace1, noisy_img1


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    # parser.add_argument('--viz', '-v', action='store_true',
    #                     help="Visualize the images as they are processed",
    #                     default=False)
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
    args.mask_path = '/HOME/reconstructed/V1/MATLAB/mask_100_140.mat'
    args.img_size = 140
    args.viz = True
    args.save = True
    # args.save_path = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_aspectTransfer_2/val_results/'
    args.save_path = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_aspectTransfer_flips/test_results/'


    args.NumInputSlices = 3
    args.masked_kspace = False
    # args.model = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_aspectTransfer_2/CP_epoch37.pth'
    args.model = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_aspectTransfer_flips/CP_epoch30.pth'

    args.input_path = '/HOME/reconstructed/data/aspect/hdf5_norm/test/'
    # args.input_path = '/HOME/reconstructed/data/aspect/hdf5_norm/val/'
    return args


# def get_output_filenames(args):
#     in_files = args.input
#     out_files = []
#
#     if not args.output:
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output):
#         logging.error("Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output
#
#     return out_files


# def mask_to_image(mask):
#     return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    # out_files = get_output_filenames(args)

    gpu_id = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logging.info(f'Using device {device}')

    net = WNet(args, masked_kspace=args.masked_kspace)


    # net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))


    net.to(device=device)

    checkpoint = torch.load(args.model, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    logging.info("Model loaded !")

    in_files = glob.glob(args.input_path + '*.hdf5')


    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        with h5py.File(fn, 'r') as f:
            kspace_0 = f['kspace_0'][:]
            kspace_1 = f['kspace_1'][:]
        pred_Im0 = np.zeros((args.img_size, args.img_size, kspace_0.shape[2]), dtype=np.float32)
        pred_K0 = np.zeros_like(pred_Im0)
        pred_Im1 = np.zeros_like(pred_Im0)
        pred_K1 = np.zeros_like(pred_Im0)

        noisy0 = np.zeros_like(pred_Im0)
        noisy1 = np.zeros_like(pred_Im0)


        # img = Image.open(fn)
        for slice in range( kspace_1.shape[2]):
            add = int(args.NumInputSlices / 2)
            if slice == 0:
                K0 = np.dstack((kspace_0[:, :, 0], kspace_0[:, :, 0:2]))
                K1 = np.dstack((kspace_1[:, :, 0], kspace_1[:, :, 0:2]))

            elif slice == kspace_1.shape[2]-1:
                K0 = np.dstack((kspace_0[:, :, (kspace_1.shape[2] - 3):(kspace_1.shape[2]-1)], kspace_0[:, :, kspace_1.shape[2]-1]))
                K1 = np.dstack((kspace_1[:, :, (kspace_1.shape[2] - 3):(kspace_1.shape[2]-1)], kspace_1[:, :, kspace_1.shape[2]-1]))

            else:
                K0 = kspace_0[:, :, slice-add:slice+add+1]
                K1 = kspace_1[:, :, slice-add:slice+add+1]


            pred_Im0[:, :, slice], pred_K0[:, :, slice], pred_Im1[:, :, slice], pred_K1[:, :, slice], \
            noisy0[:, :, slice], noisy1[:, :, slice] =\
                predict(net=net, input0=K0,  input1=K1, device=device, args=args)

        if args.save:
            os.makedirs(args.save_path, exist_ok=True)
            out_file_name = args.save_path + os.path.split(fn)[1]
            save_data(pred_Im0, pred_K0, pred_Im1, pred_K1, noisy0, noisy1, out_file_name)

            logging.info("Mask saved to {}".format(out_file_name))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_imgs(pred_Im0, pred_Im1, noisy0, noisy1)
