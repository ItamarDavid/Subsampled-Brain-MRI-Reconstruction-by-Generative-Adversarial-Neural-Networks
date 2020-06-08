import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from Generator import UNet, WNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import AspectDataset_multi, IXIataset_multi
from loss import netLoss
from torch.utils.data import DataLoader, random_split

# dir_img = 'data/imgs/'
# dir_mask = 'data/masks/'
dir_checkpoint = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_aspectTransfer_flips/'


def train_net(net, device, args):

    if args.dataset == 'Aspect':
        train_dataset = AspectDataset_multi(args.train_dir, args)
        val_dataset = AspectDataset_multi(args.val_dir, args, validtion_flag=True)
    elif args.dataset == 'IXI':
        train_dataset = IXIataset_multi(args.train_dir, args)
        val_dataset = IXIataset_multi(args.val_dir, args, validtion_flag=True)



    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True) #shuffle is true just for the diffrent images on tensorboard

    writer = SummaryWriter(log_dir=dir_checkpoint + '/runs', comment=f'LR_{args.lr}_BS_{args.batchsize}')

    logging.info(f'''Starting training:
        Epochs:          {args.epochs_n}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Checkpoints:     {args.save_cp}
        Device:          {device}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    net.to(device=device)
    start_epoch=0


    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])

        if args.load_scheduler_optimizer:
            start_epoch = int(checkpoint['epoch'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f'Model, optimizer and scheduler load from {args.load}')
        else:
            logging.info(f'Model only load from {args.load}')



    # criterion = nn.MSELoss()
    criterion = netLoss(args, masked_kspace=args.masked_kspace)


    for epoch in range(start_epoch, args.epochs_n):
        net.train()

        epoch_loss = 0
        progress_img = 0
        with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs_n}', unit=' imgs') as pbar:
            #train
            for batch in train_loader:
                masked_Kspace = batch['masked_Kspace']
                full_Kspace = batch['full_Kspace']
                full_img = batch['full_img']

                # assert imgs.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                masked_Kspace = masked_Kspace.to(device=device, dtype=torch.float32)

                full_Kspace = full_Kspace.to(device=device, dtype=torch.float32)
                full_img = full_img.to(device=device, dtype=torch.float32)

                # true_masks = true_masks.to(device=device, dtype=mask_type)

                rec_img, rec_Kspace, F_rec_Kspace = net(masked_Kspace)

                FullLoss, ImL2, ImL1, KspaceL2 = criterion.calc(rec_img, rec_Kspace, full_img, full_Kspace)
                epoch_loss += FullLoss.item()
                writer.add_scalar('train/FullLoss', FullLoss.item(), epoch)
                writer.add_scalar('train/ImL2', ImL2.item(), epoch)
                writer.add_scalar('train/ImL1', ImL1.item(), epoch)
                writer.add_scalar('train/KspaceL2', KspaceL2.item(), epoch)

                progress_img += 100*full_Kspace.shape[0]/len(train_dataset)
                pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                    'KspaceL2': KspaceL2.item(), 'Prctg of train set': progress_img})


                optimizer.zero_grad()
                FullLoss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(full_Kspace.shape[0]) #batch size
            # if epoch:
            writer.add_images('train/Fully_sampled_images', full_img, epoch)
            writer.add_images('train/rec_images', rec_img, epoch)
            writer.add_images('train/Kspace_rec_images', F_rec_Kspace, epoch)


            # writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, epoch)

            #validation:
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
            val_rec_img, val_full_img, val_F_rec_Kspace, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR =\
                eval_net(net, val_loader, criterion, device)
            writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
            writer.add_images('validation/rec_images', val_rec_img, epoch)
            writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)

            scheduler.step(val_FullLoss)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                         .format(val_FullLoss,val_ImL2, val_ImL1, val_KspaceL2, val_PSNR))
            writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
            writer.add_scalar('validation/ImL2', val_ImL2, epoch)
            writer.add_scalar('validation/ImL2', val_ImL2, epoch)
            writer.add_scalar('validation/ImL1', val_ImL1, epoch)
            writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
            writer.add_scalar('validation/PSNR', val_PSNR, epoch)




        if args.save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            # torch.save(net.state_dict(),
            #            dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Kspace recunstruction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    # parser.add_argument('-f', '--load', dest='load', type=str, default=False,
    #                     help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    args = parser.parse_args()

    args.load = '/media/rrtammyfs/Users/Itamar/reconstructed/V0_30_multi/CP_epoch56.pth' #False or path

    if args.load:
        args.load_scheduler_optimizer = False # if resume training

    args.bilinear = True

    args.dataset = 'Aspect'
    args.masked_kspace = False

    #aspect:
    # args.train_dir = '/HOME/reconstructed/data/aspect/train/'
    # args.val_dir = '/HOME/reconstructed/data/aspect/val/'
    if args.dataset == 'IXI':
        #IXI:
        args.train_dir = '/HOME/reconstructed/data/IXIhdf5/train/'
        args.val_dir = '/HOME/reconstructed/data/IXIhdf5/val/'

    if args.dataset == 'Aspect':
        #IXI:
        args.train_dir = '/HOME/reconstructed/data/aspect/hdf5_norm/train/'
        args.val_dir = '/HOME/reconstructed/data/aspect/hdf5_norm/val/'


    args.mask_path = '/HOME/reconstructed/V1/MATLAB/mask_100_140.mat'
    args.NumInputSlices = 3
    args.img_size = 140
    args.lr = 0.001
    args.epochs_n = 100
    # args.slice_range = [20, 120]


    args.LossWeights = [5, 5, 5]

    args.save_cp = True
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    gpu_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = 'cuda'
    args.device = device
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = WNet(args, masked_kspace=args.masked_kspace)
    logging.info(f'Network:\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')


    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net, device, args)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
