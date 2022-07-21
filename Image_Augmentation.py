from imgaug import augmenters as iaa
import imgaug as ia
import imgaug
import numpy as np
import torch
import os
from numpy import linalg
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


def show_batch(inputs):
    """Show image with landmarks for a batch of samples."""

    batch_size = len(inputs)
    im_size = inputs.size(2)

    grid = utils.make_grid(inputs,padding=10)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

class Color_Deconvolution(object):

    def __init__(self):
        self.params = {
            'image_type': 'HEDab'
        }

        return

    def log_transform(self, colorin):
        res = - 255.0 / np.log(256.0) * np.log((colorin + 1) / 256.0)
        res[res < 0] = 0.0
        res[res > 255.0] = 255.0
        return res

    def exp_transform(self, colorin):
        res = np.exp((255 - colorin) * np.log(255) / 255)
        res[res < 0] = 0.0
        res[res > 255.0] = 255.0
        return res

    def colorDeconv(self, imin):
        M_h_e_dab_meas = np.array([[0.650, 0.072, 0.268],
                                      [0.704, 0.990, 0.570],
                                      [0.286, 0.105, 0.776]])

        # [H,E]
        M_h_e_meas = np.array([[0.644211, 0.092789],
                                  [0.716556, 0.954111],
                                  [0.266844, 0.283111]])

        if self.params['image_type'] == "HE":
            # print "HE stain"
            M = M_h_e_meas
            M_inv = np.dot(linalg.inv(np.dot(M.T, M)), M.T)

        elif self.params['image_type'] == "HEDab":
            # print "HEDab stain"
            M = M_h_e_dab_meas
            M_inv = linalg.inv(M)

        else:
            # print "Unrecognized image type !! image type set to \"HE\" "
            M = np.diag([1, 1, 1])
            M_inv = np.diag([1, 1, 1])

        imDecv = np.dot(self.log_transform(imin.astype('float')), M_inv.T)
        imout = self.exp_transform(imDecv)

        return imout

    def colorDeconvHE(self, imin):
        """
        Does the opposite of colorDeconv
        """
        M_h_e_dab_meas = np.array([[0.650, 0.072, 0.268],
                                      [0.704, 0.990, 0.570],
                                      [0.286, 0.105, 0.776]])

        # [H,E]
        M_h_e_meas = np.array([[0.644211, 0.092789],
                                  [0.716556, 0.954111],
                                  [0.266844, 0.283111]])

        if self.params['image_type'] == "HE":
            # print "HE stain"
            M = M_h_e_meas

        elif self.params['image_type'] == "HEDab":
            # print "HEDab stain"
            M = M_h_e_dab_meas

        else:
            # print "Unrecognized image type !! image type set to \"HE\" "
            M = np.diag([1, 1, 1])
            M_inv = np.diag([1, 1, 1])

        imDecv = np.dot(self.log_transform(imin.astype('float')), M.T)
        imout = self.exp_transform(imDecv)

        return imout

class Image_Augmentation_Strong(object):
    def __init__(self, Image_Size):
        self.torch_Color = transforms.Compose([
                                transforms.Resize(Image_Size),
                                # transforms.RandomHorizontalFlip(p=0.5),
                                # transforms.RandomVerticalFlip(p=0.5),
                                # transforms.RandomResizedCrop(size=Image_Size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                                # transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.1),
                            ])
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.iaa_Geo = iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images

                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
    random_order=True
)
    def __call__(self, img):
        img = np.array(img)
        img = self.iaa_Geo.augment_image(img)
        img = Image.fromarray(img)
        img = self.torch_Color(img)
        return img


class Image_Augmentation(object):
    def __init__(self, Image_Size):
        self.torch_Color = transforms.Compose([
            #transforms.RandomCrop(Image_Size//2),
            transforms.Resize(Image_Size),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomResizedCrop(size=Image_Size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            # transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.1),
        ])
        self.iaa_Geo = iaa.Sequential([
            # iaa.Rot90(k=(0, 3)),
            # iaa.GaussianBlur(sigma=(0, 5)),
            # iaa.PiecewiseAffine(scale=(0, 0.025), nb_rows=(5, 10), nb_cols=(5, 10), order=0, cval=0, mode=imgaug.ALL),
            # iaa.Affine(translate_percent=(-0.05, 0.05), rotate=(-15, 15), shear=(-5, 5), order=0, cval=0,
            #            fit_output=False,
            #            mode=imgaug.ALL),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.iaa_Geo.augment_image(img)
        img = Image.fromarray(img)
        img = self.torch_Color(img)
        return img


if __name__ == '__main__':


    Batch_Size = 4
    Num_Workers = 0  ## Using Arontier-HYY computer, 8 cores are faster than 16 cores
    # dataset_dir = r'D:\Projects\Gastric_Cancer\Data\Gastric_Cancer\tmp\cnorm\asan'
    # dataset_dir = r'D:\Projects\Colon\Data\Samsung_Colon\Tiles\TS1024_OL512_train_samsung\Val'
    dataset_dir = r'E:\Projects\ECDP2020\Slides\HEROHE_CHALLENGE\tmp_Tile_Dicts_w_230_0.75_std_10\tmp_image_aug'

    transform = transforms.Compose([
        Image_Augmentation(240),
        # Image_Augmentation(1024),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_datasets_transformed = datasets.ImageFolder(dataset_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(img_datasets_transformed, batch_size=Batch_Size, shuffle=False, num_workers=Num_Workers)

    for i, (inputs, labels) in enumerate(dataloader):
        for n_inputs in range(inputs.size()[0]):
            plt.figure()
            plt.imshow(inputs[n_inputs].numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.ioff()
            plt.show()


        # plt.figure()
        # utils.save_image(inputs,'tmp.jpg', padding=10)
        # show_batch(inputs)
        # plt.axis('off')
        # plt.ioff()
        # plt.show()