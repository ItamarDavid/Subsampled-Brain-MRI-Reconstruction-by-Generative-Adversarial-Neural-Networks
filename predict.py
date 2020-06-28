import yaml
from types import SimpleNamespace
import logging
import os

import numpy as np
import torch
import glob
import h5py
from pprint import pprint

from Networks import WNet
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
    net_input0 = np.zeros((2*args.num_input_slices, args.img_size, args.img_size))
    net_input1 = np.zeros((2*args.num_input_slices, args.img_size, args.img_size))

    noisy_img0 = np.zeros((args.img_size, args.img_size))
    noisy_img1 = np.zeros((args.img_size, args.img_size))
    for slice in range(args.num_input_slices):
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
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)

    args.masked_kspace = True
    args.mask_path = './Masks/mask_{}_{}.pickle'.format(args.sampling_percentage, args.img_size)
    pprint(data)


    return args


if __name__ == "__main__":
    args = get_args()

    # Set device and GPU (currently only single GPU training is supported
    logging.info(f'Using device {args.device}')
    if args.device == 'cuda':
        logging.info(f'Using GPU {args.gpu_id}')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Load network
    logging.info("Loading model {}".format(args.model))
    net = WNet(args, masked_kspace=args.masked_kspace)
    net.to(device=args.device)

    checkpoint = torch.load(args.model, map_location=args.device)
    net.load_state_dict(checkpoint['G_model_state_dict'])

    logging.info("Model loaded !")


    test_files = glob.glob(os.path.join(args.predict_data_dir, '*.hdf5'))


    for i, fn in enumerate(test_files):
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
            add = int(args.num_input_slices / 2)
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
                predict(net=net, input0=K0,  input1=K1, device=args.device, args=args)

        if args.save_prediction:
            os.makedirs(args.save_path, exist_ok=True)
            out_file_name = args.save_path + os.path.split(fn)[1]
            save_data(pred_Im0, pred_K0, pred_Im1, pred_K1, noisy0, noisy1, out_file_name)

            logging.info("Mask saved to {}".format(out_file_name))

        if args.visualize_images:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_imgs(pred_Im0, pred_Im1, noisy0, noisy1)
