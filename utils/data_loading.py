from pathlib import Path
from os.path import splitext
import logging
import numpy as np
from os import listdir, path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import torchvision.transforms.functional as fn
# import os
import random
# import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FSDataset(Dataset):
    def __init__(self, dataset_dir: Path, transform=False, scale: float = 1.0):
        self.images_dir = dataset_dir / Path("imgs")
        self.masks_dir = dataset_dir / Path("masks")
        self.save_dir = "home/capurjos/"
        self.scale = scale
        self.filenames = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if not self.filenames:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.filenames)} examples')
        # self.transform = transforms.Compose([
        #         transforms.Resize((1280, 420)),
        #         transforms.ToTensor()
        #     ])

    def __len__(self):
        """
        :return: length of dataset
        """
        return len(self.filenames)


    @staticmethod
    def set_normalization_transformation(mean, variance):
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean[0], mean[1], mean[2]), 
                                                     (variance[0], variance[1], variance[2]))])


    @staticmethod
    def compute_mean_and_variance(train_subset):
        """
        computes mean and variance of *training* dataset
        :param train_images: dataloader - tensor
        :return: tuple - mean, variance
        """
        sub = [train_subset.dataset[i]["image"] for i in train_subset.indices]
        sub = torch.stack(sub)
        print(torch.sum(sub, dim=[2,3]))
        # print(torch.mean(sub, dim=[1, 2]))
        # print(torch.var(sub, dim=[1, 2]))
        return
        for i in train_subset.indices:
            img = train_subset.dataset[i]["image"]
            r, g, b = torch.mean(img, dim=[1, 2])
            print(r,g,b)
        return

        mean_train = torch.mean(train_subset.dataset[dataloader.indices]["images"], dim=0)
        std_train = torch.std(dataloader.dataset[dataloader.indices]["images"], dim=0)
        print(mean_train)
        print(std_train)
        for data in dataloader:
            data = data['image']
            data.mean()

            # Mean over batch, height and width, but not over the channels


            # print("The mean ofdef transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(520, 520))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask 
 
        
        mean = channels_sum / num_batches

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std


    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        """
        :param pil_img: input image - PIL object
        :param scale:
        :param is_mask:
        :return:
        """
        # rescale image
        # print(pil_img.size)
        # TODO uncomment, only for synthetic data!!!!
        pil_img = pil_img.crop((0, 300, 1280, 720))
        # if is_mask == False:
        #     pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((512, 384), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        w, h = pil_img.size

        # pil_img = pil_img.crop((0, 90, w, h))
        # TODO
        pil_img = pil_img.convert('RGB')
        # w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # if not is_mask:
        #     # transform = transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))
        #     # pil_img = transforms.functional.rgb_to_grayscale(pil_img, num_output_channels=1)
        #     opencv_img = np.array(pil_img)
        #     opencv_img = opencv_img[:, :, ::-1].copy() 
        #     hsv = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2HSV)
            
        #     lower_yellow = np.array([22, 93, 0], dtype="uint8")
        #     upper_yellow = np.array([45, 255, 255], dtype="uint8")
        #     mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        #     lower_blue = np.array([110,50,50])
        #     upper_blue = np.array([130,255,255])
        #     mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        #     lower_orange = np.array([0, 100, 45])
        #     upper_orange = np.array([225, 250, 255])
        #     mask_orange = cv2.inRange(hsv,lower_orange, upper_orange)
        #     merged_mask = cv2.bitwise_or(mask_yellow, mask_blue)
        #     merged_mask = cv2.bitwise_or(merged_mask, mask_orange)
        #     detected_output = cv2.bitwise_and(opencv_img, opencv_img, mask = merged_mask) 
        #     pil_img = detected_output
        #     # pil_img = transform(pil_img)
        #     folder="saved_images/"
        #     path = os.path.join(folder, str(random.randint(0, 100)) + ".png")
            # cv2.imwrite(path, detected_output)
        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.copy()
        # print(f"origin shape is {img_ndarray.shape}")
        try:
            # TODO change. dont want to use single channel images..
            if img_ndarray.shape[2] == 4 or img_ndarray.shape[2] == 1:
                pil_img = pil_img.convert('RGB')
                img_ndarray = np.asarray(pil_img)
        except:
            print("error occured when trying to convert to rgb")
        if is_mask:
            # if len(img_ndarray.shape) > 1:
            try:
                # TODO is this okay?? prob. yes
                img_ndarray = img_ndarray[:, :, 0]
                if np.all(img_ndarray == 0):
                    # print("image {0} contains only one label, not saving!".format(""))
                    return
            except:
                pass
            

        # print(f"shape is {img_ndarray.shape}")
        # img_tensor = torch.tensor(img_ndarray , dtype= torch.float)
        # # print(img_ndarray.shape)
        # if is_mask == False:
        #     image =  (img_tensor - img_tensor.mean([0, 1])) / img_tensor.std([0, 1])
        #     img_ndarray = np.asarray(image)
        # # TODO?
        if not is_mask:
            # img_ndarray[img_ndarray==255] = 1
            # TODO uncomment???
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
                # print("new axis added")
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            # TODO!!!!!
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        """
        :param filename:
        :return: single Pillow image
        """
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)





    def __getitem__(self, idx):
        # print(f"image {self.filenames[idx]} loaded")
        # print(f"current index is {idx}")
        name = self.filenames[idx]
        # print(f"loading image {name}")
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the image {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the image {name}: {mask_file}'
        mask = self.load(mask_file[0])
        image = self.load(img_file[0])

        assert image.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {image.size} and {mask.size}'
        # TODO?? 3, 640, 959
        img = self.preprocess(image, self.scale, is_mask=False)
        # print(f"shape of input image is {img.shape}")
        # # 640, 959
        mask = self.preprocess(mask, self.scale, is_mask=True)
        # print(f"shape of input mask is {mask.shape}")
        # print(img.shape)
        # assert img.shape[0] == 420

        sample = {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
        # print(f"loading image {name} was successful")
    
        return sample