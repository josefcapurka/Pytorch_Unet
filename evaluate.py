import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from random import randint


from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device, experiment):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    i = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        # wandb.Image(mask_true.float().cpu()),
        # wandb.Image(image.cpu())
        i += 1

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            # wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu())

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            for k in range(10):
                #if k < 3:
                if i % randint(1, 20) == 0:
                    experiment.log({
                                    'images': wandb.Image(image[0].float().cpu()),
                                    'masks': {
                                        'true': wandb.Image(mask_true[0].float().cpu()),
                                        'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                                    },

                                })

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
