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
from Generator import WNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import IXIataset #, AspectDataset
from loss import netLoss
from torch.utils.data import DataLoader, random_split



dir_checkpoint = '/media/rrtammyfs/Users/Itamar/reconstructe/SameAS/'


def train_nets(gen_net, gen_optimizer, gen_scheduler, args):

    # if args.dataset == 'Aspect':
    #     train_dataset = AspectDataset(args.train_dir, args)
    #     val_dataset = AspectDataset(args.val_dir, args, validtion_flag=True)

    if args.dataset == 'IXI':
        train_dataset = IXIataset(args.train_dir, args)
        val_dataset = IXIataset(args.val_dir, args, validtion_flag=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True) #shuffle is true just for the diffrent images on tensorboard
    
    #TODO: better name for checkpoints dir
    writer = SummaryWriter(log_dir=dir_checkpoint + '/runs', comment=f'LR_{args.lr}_BS_{args.batchsize}')

    logging.info(f'''Starting training:
        Epochs:          {args.epochs_n}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Checkpoints:     {args.save_cp}
        Device:          {args.device}
    ''')
 
    gen_net.to(device=device)
    start_epoch = 0

    if args.load:
        checkpoint = torch.load(args.load, map_location=args.device)
        gen_net.load_state_dict(checkpoint['model_state_dict'])

        if args.load_scheduler_optimizer:
            start_epoch = int(checkpoint['epoch'])
            gen_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            gen_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f'Model, optimizer and scheduler load from {args.load}')
        else:
            logging.info(f'Model only load from {args.load}')
            
    criterion = netLoss(args)

    for epoch in range(start_epoch, args.epochs_n):
        gen_net.train()
        epoch_loss = 0
        progress_img = 0
        with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs_n}', unit=' imgs') as pbar:
            #train
            for batch in train_loader:

                masked_Kspaces = batch['masked_Kspaces']
                target_Kspace = batch['target_Kspace']
                target_img = batch['target_img']

                masked_Kspaces = masked_Kspaces.to(device=args.device, dtype=torch.float32)
                target_Kspace = target_Kspace.to(device=args.device, dtype=torch.float32)
                target_img = target_img.to(device=args.device, dtype=torch.float32)

                rec_img, rec_Kspace, F_rec_Kspace = gen_net(masked_Kspaces)

                FullLoss, ImL2, ImL1, KspaceL2 = criterion.calc(rec_img, rec_Kspace, target_img, target_Kspace)
                
                epoch_loss += FullLoss.item()
                writer.add_scalar('train/FullLoss', FullLoss.item(), epoch)
                writer.add_scalar('train/ImL2', ImL2.item(), epoch)
                writer.add_scalar('train/ImL1', ImL1.item(), epoch)
                writer.add_scalar('train/KspaceL2', KspaceL2.item(), epoch)

                progress_img += 100*target_Kspace.shape[0]/len(train_dataset)
                pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                    'KspaceL2': KspaceL2.item(), 'Prctg of train set': progress_img})

                gen_optimizer.zero_grad()
                FullLoss.backward()
                #TODO: Do we need this clipping?
                nn.utils.clip_grad_value_(gen_net.parameters(), 0.1)
                gen_optimizer.step()

                pbar.update(target_Kspace.shape[0])# current batch size

            # if epoch:
            writer.add_images('train/Fully_sampled_images', target_img, epoch)
            writer.add_images('train/rec_images', rec_img, epoch)
            writer.add_images('train/Kspace_rec_images', F_rec_Kspace, epoch)

            for tag, value in gen_net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

            # validation:
            val_rec_img, val_full_img, val_F_rec_Kspace, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR =\
                eval_net(gen_net, val_loader, criterion, args.device)
            gen_scheduler.step(val_FullLoss)

            writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
            writer.add_images('validation/rec_images', val_rec_img, epoch)
            writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)

            writer.add_scalar('learning_rate', gen_optimizer.param_groups[0]['lr'], epoch)

            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR))
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
                'model_state_dict': gen_net.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
                'scheduler_state_dict': gen_scheduler.state_dict(),
            }, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()


def get_args():
    #TODO: fix arguments loader (YAML?)
    parser = argparse.ArgumentParser(description='Kspace recunstruction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')

    args = parser.parse_args()

    args.load = False #False or path
    args.resume = False

    if args.resume:
        args.load_scheduler_optimizer = True# if resume training

    args.bilinear = True

    args.dataset = 'IXI'
    # if args.dataset == 'aspect':
        #aspect:
        # args.train_dir = '/HOME/reconstructed/data/aspect/train/'
        # args.val_dir = '/HOME/reconstructed/data/aspect/val/'
    if args.dataset == 'IXI':
        #IXI:
        args.train_dir = '/HOME/reconstructed/data/IXIhdf5/train/'
        args.val_dir = '/HOME/reconstructed/data/IXIhdf5/val/'


    args.mask_path = '/HOME/reconstructed/SameAs/MATLAB/masks/mask_30_256.mat'
    args.NumInputSlices = 3
    args.img_size = 256
    args.lr = 0.001
    args.epochs_n = 100
    args.slice_range = [20, 120]


    args.LossWeights = [5, 5, 5, ] #Imspac L2, Imspace L1, Kspace L2, GAN_Loss

    args.save_cp = True
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    gpu_id = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = 'cuda'
    
    args.device = device
    
    logging.info(f'Using device {device}')

    gen_net = WNet(args)
    logging.info(f'Network:\n'
                 f'\t{"Bilinear" if gen_net.bilinear else "Transposed conv"} upscaling')
    gen_optimizer = optim.RMSprop(gen_net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, 'min', patience=2)

    #TODO:
    # dis_net = Discriminator(args)
    # dis_optimizer = optim.RMSprop(dis_net.parameters() + gen_net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    # dis_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dis_optimizer, 'min', patience=2)


    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_nets(gen_net, gen_optimizer, gen_scheduler, args)

    except KeyboardInterrupt:
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save({
            'epoch': '',
            'model_state_dict': gen_net.state_dict(),
            'optimizer_state_dict': gen_optimizer.state_dict(),
            'scheduler_state_dict': gen_scheduler.state_dict(),
        }, dir_checkpoint + f'INTERRUPTED.pth')

        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
