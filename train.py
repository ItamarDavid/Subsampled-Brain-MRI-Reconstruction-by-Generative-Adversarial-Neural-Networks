import argparse
import logging
import os
import sys

import torch
from torch import optim
from tqdm import tqdm

from eval import eval_net
from Networks import WNet,NLayerDiscriminator
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import IXIdataset
from loss import netLoss,set_grad
from torch.utils.data import DataLoader



def train_nets(gen_net, gen_optimizer, gen_scheduler,dis_net,dis_optimizer, args):


    if args.dataset == 'IXI':
        train_dataset = IXIdataset(args.train_dir, args)
        val_dataset = IXIdataset(args.val_dir, args, validtion_flag=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True) #Shuffle is true for diffrent images on tensorboard
    
    writer = SummaryWriter(log_dir=args.dir_checkpoint + '/runs', comment=f'LR_{args.lr}_BS_{args.batchsize}')

    logging.info(f'''Starting training:
        Epochs:          {args.epochs_n}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Checkpoints:     {args.save_cp}
        Device:          {args.device}
    ''')
 
    gen_net.to(device=device)
    dis_net.to(device=device)
    start_epoch = 0

    if args.load_gen:
        checkpoint = torch.load(args.load_gen, map_location=args.device)
        gen_net.load_state_dict(checkpoint['model_state_dict'])

        if args.load_scheduler_optimizer:
            start_epoch = int(checkpoint['epoch'])
            gen_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            gen_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f'Generator Model, optimizer and scheduler load from {args.load_gen}')
        else:
            logging.info(f'Generator Model only load from {args.load_gen}')

    if args.load_dis:
        checkpoint = torch.load(args.load_dis, map_location=args.device)
        dis_net.load_state_dict(checkpoint['model_state_dict'])

        if args.load_scheduler_optimizer:
            start_epoch = int(checkpoint['epoch'])
            dis_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f'Discriminator Model, optimizer and scheduler load from {args.load_dis}')
        else:
            logging.info(f'Discriminator Model only load from {args.load_dis}')
            
    criterion = netLoss(args)

    for epoch in range(start_epoch, args.epochs_n):
        gen_net.train()
        dis_net.train()
        epoch_gen_loss = 0
        epoch_dis_loss = 0
        progress_img = 0
        with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs_n}', unit=' imgs') as pbar:
            #Train loop
            for batch in train_loader:

                masked_Kspaces = batch['masked_Kspaces']
                target_Kspace = batch['target_Kspace']
                target_img = batch['target_img']

                masked_Kspaces = masked_Kspaces.to(device=args.device, dtype=torch.float32)
                target_Kspace = target_Kspace.to(device=args.device, dtype=torch.float32)
                target_img = target_img.to(device=args.device, dtype=torch.float32)

                rec_img, rec_Kspace, F_rec_Kspace = gen_net(masked_Kspaces)
                real_D_ex = target_img.detach()
                fake_D_ex = rec_img
                D_real = dis_net(real_D_ex)
                D_fake = dis_net(fake_D_ex)
                FullLoss, ImL2, ImL1, KspaceL2, advLoss = criterion.calc_gen_loss(rec_img, rec_Kspace, target_img, target_Kspace, D_fake)

                #Stop backprop to G by detaching

                D_fake_detach = dis_net(fake_D_ex.detach())
                D_real_loss,D_fake_loss,DLoss = criterion.calc_disc_loss(D_real, D_fake_detach)


                epoch_gen_loss += FullLoss.item()
                epoch_dis_loss += DLoss.item()

                progress_img += 100*target_Kspace.shape[0]/len(train_dataset)
                train_D = advLoss.item()<D_real_loss.item()*1.5
                pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                    'KspaceL2': KspaceL2.item(),'Adv G': advLoss.item(),'Adv D - Real' : D_real_loss.item(),'Adv D - Fake' : D_fake_loss.item(),'Train D': train_D, 'Prctg of train set': progress_img})

                # torch.autograd.set_detect_anomaly(True)
                #Optimize parameters
                #Update G
                set_grad(dis_net, False)  # No D update
                gen_optimizer.zero_grad()
                FullLoss.backward()
                gen_optimizer.step()
                #Update D
                set_grad(dis_net, True)  # enable backprop for D
                if train_D:
                    dis_optimizer.zero_grad()
                    DLoss.backward()
                    dis_optimizer.step()



                pbar.update(target_Kspace.shape[0])# current batch size

            # On epoch end
            # Validation
            val_rec_img, val_full_img, val_F_rec_Kspace, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR =\
                eval_net(gen_net, val_loader, criterion, args.device)
            gen_scheduler.step(val_FullLoss)


            writer.add_scalar('train/FullLoss', FullLoss.item(), epoch)
            writer.add_scalar('train/ImL2', ImL2.item(), epoch)
            writer.add_scalar('train/ImL1', ImL1.item(), epoch)
            writer.add_scalar('train/KspaceL2', KspaceL2.item(), epoch)
            writer.add_scalar('train/G_AdvLoss', advLoss.item(), epoch)
            writer.add_scalar('train/D_AdvLoss', DLoss.item(), epoch)
            writer.add_images('train/Fully_sampled_images', target_img, epoch)
            writer.add_images('train/rec_images', rec_img, epoch)
            writer.add_images('train/Kspace_rec_images', F_rec_Kspace, epoch)
            writer.add_scalar('train/learning_rate', gen_optimizer.param_groups[0]['lr'], epoch)

            writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
            writer.add_images('validation/rec_images', val_rec_img, epoch)
            writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)
            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR))
            writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
            writer.add_scalar('validation/ImL2', val_ImL2, epoch)
            writer.add_scalar('validation/ImL1', val_ImL1, epoch)
            writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
            writer.add_scalar('validation/PSNR', val_PSNR, epoch)




        if args.save_cp:
            try:
                os.mkdir(args.dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save({
                'epoch': epoch,
                'model_state_dict': gen_net.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
                'scheduler_state_dict': gen_scheduler.state_dict(),
            }, args.dir_checkpoint + f'G_CP_epoch{epoch + 1}.pth')

            torch.save({
                'epoch': epoch,
                'model_state_dict': dis_net.state_dict(),
                'optimizer_state_dict': dis_optimizer.state_dict(),
            }, args.dir_checkpoint + f'D_CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()


def get_args():
    #TODO: fix arguments loader (YAML?)
    parser = argparse.ArgumentParser(description='Kspace recunstruction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=24,
                        help='Batch size', dest='batchsize')

    args = parser.parse_args()
    # args.dir_checkpoint = '/HOME/published_code/run_0/'
    args.dir_checkpoint = '/media/rrtammyfs/Users/Itamar/reconstructed/try'


    args.load_gen = False#r"/HOME/published_code/runs/G_CP_epoch3.pth" #False or path
    args.load_dis = False#r"/HOME/published_code/runs/D_CP_epoch3.pth"
    args.resume = False

    if args.resume:
        args.load_scheduler_optimizer = False# if resume training

    args.bilinear = True

    args.dataset = 'IXI'
    if args.dataset == 'IXI':
        #IXI:
        # args.train_dir = '/HOME/published_code/data/train/'
        # args.val_dir = '/HOME/published_code/data/val/'
        args.train_dir = '/HOME/reconstructed/data/IXIhdf5/train/'
        args.val_dir = '/HOME/reconstructed/data/IXIhdf5/val/'


    args.sampling_percentage = 30 #20%, 30% or 50%
    args.mask_path = './Masks/mask_{}_256.pickle'.format(args.sampling_percentage)
    args.NumInputSlices = 3
    args.img_size = 256
    args.lr = 0.001
    args.epochs_n = 100
    args.slice_range = [20, 120]


    args.LossWeights = [1000, 1000, 5, 0.1 ] #Imspac L2, Imspace L1, Kspace L2, GAN_Loss

    args.save_cp = True
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    gpu_id = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = 'cuda'
    
    args.device = device

    logging.info(f'Using device {device}')
    # Generator network
    gen_net = WNet(args)
    logging.info(f'Network:\n'
                 f'\t{"Bilinear" if gen_net.bilinear else "Transposed conv"} upscaling')
    gen_optimizer = torch.optim.Adam(gen_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, 'min', patience=5)

    #Discriminator network
    dis_net = NLayerDiscriminator(1, crop_center=(128,128))
    dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr=2*args.lr, betas=(0.5, 0.999))


    try:
        train_nets(gen_net, gen_optimizer, gen_scheduler,dis_net,dis_optimizer, args)

    except KeyboardInterrupt:
        try:
            os.mkdir(args.dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save({
            'epoch': '',
            'model_state_dict': gen_net.state_dict(),
            'optimizer_state_dict': gen_optimizer.state_dict(),
            'scheduler_state_dict': gen_scheduler.state_dict(),
        }, args.dir_checkpoint + f'gen_INTERRUPTED.pth')
        torch.save({
            'epoch': '',
            'model_state_dict': dis_net.state_dict(),
            'optimizer_state_dict': dis_optimizer.state_dict(),
        }, args.dir_checkpoint + f'dis_INTERRUPTED.pth')

        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
