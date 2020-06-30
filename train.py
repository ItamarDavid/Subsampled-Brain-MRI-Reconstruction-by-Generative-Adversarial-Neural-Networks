import yaml
from types import SimpleNamespace
import logging
import os
import sys
import shutil

import torch
from tqdm import tqdm
from pprint import pprint

from eval import eval_net
from Networks import WNet,PatchGAN
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import IXIdataset
from loss import netLoss,set_grad
from torch.utils.data import DataLoader



def train(args):
    # Init Generator network
    G_model = WNet(args)
    logging.info(f'Network:\n'
                 f'\t{"Bilinear" if G_model.bilinear else "Transposed conv"} upscaling')
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience=5)
    G_model.to(device=args.device)
    # Init Discriminator network
    D_model = PatchGAN(1, crop_center=args.crop_center)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=2 * args.lr, betas=(0.5, 0.999))
    D_model.to(device=args.device)

    # Init Dataloaders
    if args.dataset == 'IXI':
        train_dataset = IXIdataset(args.train_data_dir, args)
        val_dataset = IXIdataset(args.val_data_dir, args, validtion_flag=True)
    else:
        logging.error("Data type not supported")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.val_num_workers,
                            pin_memory=True, drop_last=True) #Shuffle is true for diffrent images on tensorboard

    # Init tensorboard writer
    if args.tb_write_losses or args.tb_write_images :
        writer = SummaryWriter(log_dir=args.output_dir + '/tensorboard')

    # Init loss object
    loss = netLoss(args)

    # Load checkpoint
    if args.load_cp:
        checkpoint = torch.load(args.load_cp, map_location=args.device)
        G_model.load_state_dict(checkpoint['G_model_state_dict'])
        D_model.load_state_dict(checkpoint['D_model_state_dict'])

        if args.resume_training:
            start_epoch = int(checkpoint['epoch'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
            G_scheduler.load_state_dict(checkpoint['G_scheduler_state_dict'])
            logging.info(f'Models, optimizer and scheduler loaded from {args.load_cp}')
        else:
            logging.info(f'Models only load from {args.load_cp}')
    else:
        start_epoch = 0
    #Start training

    logging.info(f'''Starting training:
        Epochs:          {args.epochs_n}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')


    try: #Catch keyboard interrupt and save state
        for epoch in range(start_epoch, args.epochs_n):
            G_model.train()
            D_model.train()
            progress = 0

            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs_n}', unit=' imgs') as pbar:
                #Train loop
                for batch in train_loader:

                    masked_Kspaces = batch['masked_Kspaces'].to(device=args.device, dtype=torch.float32)
                    target_Kspace = batch['target_Kspace'].to(device=args.device, dtype=torch.float32)
                    target_img = batch['target_img'].to(device=args.device, dtype=torch.float32)

                    #Forward G:
                    rec_img, rec_Kspace, rec_mid_image = G_model(masked_Kspaces)

                    #Forward D for G loss:
                    if args.GAN_training:
                        real_D_example = target_img.detach()
                        fake_D_example = rec_img
                        D_real_pred = D_model(real_D_example)
                        D_fake_pred = D_model(fake_D_example)
                    else:
                        D_fake_pred = None

                    #Calc G losses:
                    FullLoss, ImL2, ImL1, KspaceL2, advLoss = loss.calc_gen_loss(rec_img, rec_Kspace, target_img, target_Kspace, D_fake_pred)

                    #Forward D for D loss:
                    if args.GAN_training:
                        D_fake_detach = D_model(fake_D_example.detach())   #Stop backprop to G by detaching
                        D_real_loss,D_fake_loss,DLoss = loss.calc_disc_loss(D_real_pred, D_fake_detach)
                        # Train/stop Train D criteria
                        train_D = advLoss.item()<D_real_loss.item()*1.5

                    #Optimize parameters
                    #Update G
                    if args.GAN_training:
                        set_grad(D_model, False)  # No D update

                    G_optimizer.zero_grad()
                    FullLoss.backward()
                    G_optimizer.step()
                    #Update D
                    if args.GAN_training:
                        set_grad(D_model, True)  # enable backprop for D
                        if train_D:
                            D_optimizer.zero_grad()
                            DLoss.backward()
                            D_optimizer.step()

                    #Update progress bar
                    progress += 100*target_Kspace.shape[0]/len(train_dataset)
                    if args.GAN_training:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(),'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                            'KspaceL2': KspaceL2.item(),'Adv G': advLoss.item(),'Adv D - Real' : D_real_loss.item(),'Adv D - Fake' : D_fake_loss.item(),'Train D': train_D, 'Prctg of train set': progress})
                    else:
                        pbar.set_postfix(**{'FullLoss': FullLoss.item(), 'ImL2': ImL2.item(), 'ImL1': ImL1.item(),
                                            'KspaceL2': KspaceL2.item(), 'Prctg of train set': progress})
                    pbar.update(target_Kspace.shape[0])# current batch size

            # On epoch end
            # Validation
            val_rec_img, val_full_img, val_F_rec_Kspace, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR =\
                eval_net(G_model, val_loader, loss, args.device)
            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR))
            # Schedular update
            G_scheduler.step(val_FullLoss)

            #Write to TB:
            if args.tb_write_losses:
                writer.add_scalar('train/FullLoss', FullLoss.item(), epoch)
                writer.add_scalar('train/ImL2', ImL2.item(), epoch)
                writer.add_scalar('train/ImL1', ImL1.item(), epoch)
                writer.add_scalar('train/KspaceL2', KspaceL2.item(), epoch)
                writer.add_scalar('train/learning_rate', G_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss', advLoss.item(), epoch)
                    writer.add_scalar('train/D_AdvLoss', DLoss.item(), epoch)

                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL2', val_ImL2, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
                writer.add_scalar('validation/PSNR', val_PSNR, epoch)

            if args.tb_write_images:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/rec_images', rec_img, epoch)
                writer.add_images('train/Kspace_rec_images', rec_mid_image, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)
                writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)

            #Save Checkpoint
            torch.save({
                'epoch': epoch,
                'G_model_state_dict': G_model.state_dict(),
                'G_optimizer_state_dict': G_optimizer.state_dict(),
                'G_scheduler_state_dict': G_scheduler.state_dict(),
                'D_model_state_dict': D_model.state_dict(),
                'D_optimizer_state_dict': D_optimizer.state_dict(),
            }, args.output_dir + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'G_model_state_dict': G_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': G_scheduler.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
        }, args.output_dir + f'CP_epoch{epoch + 1}_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    writer.close()


def get_args():
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)

    args.mask_path = './Masks/mask_{}_{}.pickle'.format(args.sampling_percentage,args.img_size)
    pprint(data)
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # Create output dir
    try:
        os.mkdir(args.output_dir)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

    # Copy configuration file to output directory
    shutil.copyfile('config.yaml',os.path.join(args.output_dir,'config.yaml'))

    # Set device and GPU (currently only single GPU training is supported
    logging.info(f'Using device {args.device}')
    if args.device == 'cuda':
        logging.info(f'Using GPU {args.gpu_id}')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    train(args)


