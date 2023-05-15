import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# from torch.autograd import Variable

from utils.data_loading import FSDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import os
from torchvision import transforms
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# dataset_dir = Path('/home/capurjos/Pytorch-UNet/cropped_imgs_raw')
# dataset_dir = Path('/home/capurjos/data')

# TODO https://stackoverflow.com/questions/61808965/pytorch-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-ha
torch.set_grad_enabled(True)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'validation_loss'
        },
    'parameters': {
        'batch_size': {'values': [1, 2, 4, 8, 16, 32, 64]},
        'epochs': {'values': [20, 30, 40]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-3,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.6,
              amp: bool = False,
              synthetic:bool = False,
              pretrained:str = "",
              test_optimizers:bool = False):

    # 1. Create dataset
    if args.synthetic:
        dataset_dir = Path('/home/capurjos/unet_dataset/synthetic_dataset')
        dir_checkpoint = Path('./synthetic_checkpoints/')
    else:
        dataset_dir = Path('/home/capurjos/big_dataset_w_fsoco')
        dir_checkpoint = Path('./real_world_checkpoints/')
        # dataset_dir = Path('/home/capurjos/big_dataset_without_czech')
    dataset = FSDataset(dataset_dir=dataset_dir, scale=img_scale, synthetic=synthetic)

    # 2. split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # generator - random number generator
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(247))
    # train_mean, train_variance = FSDataset.compute_mean_and_variance(train_subset=train_set)
    # print(train_set.mean())
    # FSDataset.set_normalization_transformation(train_mean, train_variance)
    # 3. Create Pytorch data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # print(train_loader[:,:,0])
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # print("test--------------------")
    if test_optimizers:
        momentum = 0.999
        optimizers = [#optim.Adagrad(net.parameters(), lr=0.01),
                        # optim.Adadelta(net.parameters(), lr=1),
                        # optim.Adam(net.parameters(), lr=learning_rate),
                        optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum = momentum)]
                        # optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)]
    # (Initialize logging)
    else:
        optimizers = [optim.Adam(net.parameters(), lr=1e-3)]
    for optimizer in optimizers:
        print(f"Optimizer: {optimizer.__class__.__name__}")
        experiment = wandb.init(project='U-Net', resume=False, anonymous='must')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                    val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                    amp=amp))
        print(f"batch size is {batch_size}")
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        # RMSprop is also adaptive optimizer
        # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        # hyperparameters used from here https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        #optimizer = optim.Adam(net.parameters(), lr=1e-3) #, betas=[0.9, 0.999], eps=1e-8)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
        # TODO - scheduler from https://github.com/amirhosseinh77/UNet-AerialSegmentation/blob/main/train.py
        scaler = torch.cuda.amp.GradScaler()
        # TODO gradient clipping
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        # https://wandb.ai/wandb_fc/tips/reports/How-to-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
        # scaling factor for gradient
        # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        # weights = [1.0, 7.0]
        # class_weights = torch.FloatTensor(weights).cuda()
        #loss_function = nn.BCECrossEntropyLoss()#weight=class_weights)
        # loss_function = nn.CrossEntropyLoss()
        # loss_function = ErosionDilationPenaltyLoss(0.1, 3)
        loss_function = nn.BCEWithLogitsLoss()
        # loss_function = mIoULoss(n_classes=2).to(device)
        # loss_function = FocalLoss(gamma=3/4).to(device)
        # loss_function = nn.BCELoss()
        # loss_function = nn.MSELoss()
        # loss_function = nn.NLLLoss()
        # loss_function = nn.HingeEmbeddingLoss()
        # loss_function = nn.L1Loss()
        global_step = 0

        # 5. Begin training
        for epoch in range(1, epochs+1):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images = batch['image']
                    true_masks = batch['mask']

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32)
                    # x = torch.tensor([0,1])
                    true_masks = true_masks.to(device=device)
                    # forward pass
                    with torch.cuda.amp.autocast(enabled=amp):
                        true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                        masks_pred = net(images)
                        # _, masks_pred = torch.max(masks_pred, dim=1).to_float()
                        # TODO?  https://stackoverflow.com/questions/57798033/valueerror-target-size-torch-size16-must-be-the-same-as-input-size-torch
                        # _, masks_pred = torch.max(masks_pred, 1)
                        # TODO
                        # print(masks_pred[:, :, 84:, :].shape[2])
                        # bottom_img_loss = loss_function(masks_pred[:, :, 84:, :], true_masks[:, :, 84:, :]) * 3
                        # upper_img_loss = loss_function(masks_pred[:, :, 0:84, :], true_masks[:, :, 0:84, :])
                        # print(masks_pred.shape)
                        loss = loss_function(masks_pred, true_masks)
                            #    + dice_loss(F.softmax(masks_pred, dim=1).float(),
                            #                F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                            #                multiclass=True)
                        # loss = bottom_img_loss + upper_img_loss
                        # loss = piecewise_loss(masks_pred.squeeze(1), true_masks.float())
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # loss += dice_loss(masks_pred, true_masks, multiclass=False)
                        # print("loss on training data is: {0}".format(loss)) #\
                        # + dice_loss(F.softmax(masks_pred, dim=1).float(),
                        #         F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                        #         multiclass=False)

                    optimizer.zero_grad()#set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()
                    # a = grad_scaler.scale(loss)
                    # a.require_grad = True
                    # a.backward()
                    # grad_scaler.step(optimizer)
                    # grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    # TODO??
                    # Evaluation round
                    division_step = (n_train // (10 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            histograms = {}
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                if not torch.isinf(value).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not torch.isinf(value.grad).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = evaluate(net, val_loader, device, experiment, loss_function)
                            # TODO
                            # scheduler.step(val_score)
                            logging.info('Validation Dice score: {}'.format(val_score))
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                # 'images': wandb.Image(images[0].cpu()),
                                # 'masks': {
                                #     'true': wandb.Image(true_masks[0].float().cpu()),
                                #     'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                # },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
        wandb.finish()
        # RESET parameters
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        net.to(device=device)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--synthetic', action='store_true', default=False, help='Train on synthetic dataset')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained weights from synthetic dataset')
    parser.add_argument('--test_optimizers', '-t', action='store_true', default=False, help='Evaluate impact of different optimizers')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if str(device) != 'cuda':
        print("Error, cuda is not available")
        exit(0)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  test_optimizers=args.test_optimizers)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise