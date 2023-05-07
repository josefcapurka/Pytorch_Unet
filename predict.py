import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.data_loading import FSDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import cv2
import time
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                evaluate_accuracy=False):
    net.eval()
    img = torch.from_numpy(FSDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
            # print("using softmax")
        else:
            # print("using sigmoid")
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
    # print(out_threshold)

    return (full_mask > out_threshold).numpy()
    
    if net.n_classes == 1:
        print("ttt")
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    # parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument("--directory", action="store", dest="directory", default="", required=True)
    parser.add_argument("--evaluate_accuracy", '-e', action='store_true')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def iou_numpy(outputs, labels):
    intersection = np.logical_and(outputs, labels)
    union = np.logical_or(outputs, labels)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
if __name__ == '__main__':
    args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)
    directory = args.directory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    print(f'Loading model {args.model}')
    print(f'Using device {device}')
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info(f'Model {args.model} loaded!')
    try:
        os.mkdir(directory + "/" + "masks")
    except FileExistsError:
        print("folder {0} already exists".format(directory + "/" + "masks"))
        # exit(1)

    print("cropping")

    size = (512, 384)
    print(f"Resizing masks to size {size}")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    width, height = 4000, 3000

    # Create the output video writer
    out = cv2.VideoWriter("output_test.avi", fourcc, 5.0, (width, height))
    frames_processed = 0
    total_time = 0
    # TODO duplicate
    iou = 0
    tf = transforms.Compose([
            transforms.Resize((size[1], size[0]), InterpolationMode.NEAREST),
            # transforms.ToTensor()
        ])
    for i, filename in enumerate(os.listdir(directory)):
        # if i > 1000:
        print(filename)
        #     break
        if filename.endswith(".png") or filename.endswith(".jpg") == False:
            continue
        # filename = str(i) + ".png"
    # for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        try:
            img = Image.open(directory + "/" + filename)
            # print(f"filename {filename} opened")
        except:
            print(f"cannot open file {filename}")
            continue
        # img = img.crop((0, 400, 1280, 720))
        # print(img.size)
        # TODO bicubic or nearest
        img = img.resize(size, Image.BICUBIC)
        img_arr = np.array(img, np.uint8)
        img_arr = img_arr[:, :, :3]

        img = Image.fromarray(img_arr)
        # print(args.mask_treshold)
        ##########
        start_time = time.time()
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        end_time = time.time()
        total_time += end_time - start_time
        ##########
        frames_processed += 1

        if not args.no_save:
            out_filename = directory + "/" + "masks" + "/" + filename
            result = mask_to_image(mask)
            # print(result)
            # result = result.resize((1280, 320), Image.BICUBIC)
            # TODO bicubic or nearest??
            result = result.resize((width, height), Image.NEAREST)

            if args.evaluate_accuracy:
                true_masks_dir = os.path.join(os.path.join(directory, '..'), "masks")
                # print(f"filename is {os.path.join(true_masks_dir, filename)}")
                mask_true = Image.open(os.path.join(true_masks_dir, filename)).convert('L')
                # print(np.unique(mask_true))
    
                mask_true = tf(mask_true)
                mask_true = np.array(mask_true)
                mask_true = torch.tensor(mask_true).to(torch.int64)
                m = mask_true.cpu().numpy()
                iou += iou_numpy(np.argmax(mask, axis=0), m)
                # if iou < 0.8:
                #     print(f"image {filename} has iou only {iou}")

# # Show the numpy array as an image
#                 ax.imshow(m, cmap='gray')

#                 # Save the figure to a file
#                 plt.savefig(filename + '_true.png')
#                 ax.imshow(np.argmax(mask, axis=0), cmap='gray')
#                 plt.savefig(filename + "_predicted.png")
#                 predicted_mask = torch.from_numpy(mask).to(torch.float)
       
#                 mask_true = F.one_hot(mask_true, net.n_classes).permute(2, 0, 1).float()
  
#                 assert mask_true.shape == predicted_mask.shape
#                 dice_score = dice_coeff(predicted_mask, mask_true, reduce_batch_first=False)
#                 if dice_score > 0.5:
#                     print(f"Dice score is {dice_score}")

            # result.save(out_filename)
            opacity = 0.5

# Add the mask to the original image with some opacity
            img = Image.open(directory + "/" + filename).convert("RGB")
            # img = img.crop((0, 400, 1280, 720))
            img_arr = np.array(img, np.uint8)
            # img_arr = img_arr.transpose(1, 0, 2)
            result = np.array(result.convert("RGB"))
            # print(f"img_arr shape is {img_arr.shape}, result mask shape is {result.shape}")

            try:
                overlay = cv2.addWeighted(img_arr, 1-opacity, np.array(result), opacity, 0)
            except:
                print("opencv error, fix")
                continue
            label_color = (0, 255, 0) 
            # overlay = cv2.addWeighted(np.array(result), opacity, label_color, 1 - opacity, 0)
            rgb_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_filename, rgb_image)
            out.write(rgb_image)
            logging.info(f'Mask saved to {result}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
    out.release()

    avg_time_per_frame = total_time / frames_processed
    fps = 1 / avg_time_per_frame
    fps2 = frames_processed / total_time
    print("FPS:", fps)
    print(f"FPS: {fps2}")
    print(f"Average IoU is {iou / frames_processed}")
    # |=========================================+======================+======================|
    # |   0  Tesla V100-SXM2-32GB            On | 00000000:8A:00.0 Off |                    0 |
    # | N/A   29C    P0               43W / 300W|      0MiB / 32768MiB |      0%      Default |
    # |                                         |                      |                  N/A |
    # FPS: 66.19824269409075
