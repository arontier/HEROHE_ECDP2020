import sys
import os
import numpy as np
import argparse
import random
import openslide
import torch
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_curve, auc
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataloader import DataLoader


class Grid_Dataset(torch.utils.data.Dataset):
    def __init__(self, Grid_List, Slide_Path, Tile_Size, Transform=None):
        self.Grid_List = Grid_List
        self.Transform = Transform
        self.Slide_P = openslide.OpenSlide(Slide_Path)
        self.Tile_Size = Tile_Size

    def __getitem__(self, index):
        Grid_Point = self.Grid_List[index]

        img = self.Slide_P.read_region(Grid_Point, 0, (self.Tile_Size, self.Tile_Size)).convert('RGB')

        if self.Transform is not None:
            img = self.Transform(img)

        return img

    def __len__(self):
        return len(self.Grid_List)


def BG_Model_Filter(Slide_Grid, Slide_Path, Tile_Size, BG_Model, BG_Trans, Tissue_Prob_Thres=0.5, BG_Batch_Size=32,
                    BG_Num_Workers=4):
    BG_Grid_Dataset = Grid_Dataset(Slide_Grid, Slide_Path, Tile_Size, BG_Trans)
    BG_Loader = DataLoader(BG_Grid_Dataset, batch_size=BG_Batch_Size, shuffle=False, num_workers=BG_Num_Workers,
                           pin_memory=False)
    Tissue_Probs = inference(BG_Loader, BG_Model)
    Slide_Grid = np.array(Slide_Grid)
    Filtered_Slide_Grid = Slide_Grid[Tissue_Probs > Tissue_Prob_Thres].tolist()

    return Filtered_Slide_Grid


def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Background Inference Batch: [{}/{}]'.format(i + 1, len(loader)))
            input = input.to(device)
            output = F.softmax(model(input), dim=1)
            probs[i * loader.batch_size:i * loader.batch_size + input.size(0)] = output.detach()[:, 1].clone()
            # print(output.cpu().numpy())
    return probs.cpu().numpy()


def Make_Grid(Slide_Path, Level, Tile_Size, Overlap, White_BG_Thres):
    Grid = []
    Propose_Downsample_Times = 16
    Slide_P = openslide.OpenSlide(Slide_Path)
    Closest_Level_in_The_Slide = Slide_P.get_best_level_for_downsample(Propose_Downsample_Times + 1)

    Downsample_Level = Closest_Level_in_The_Slide
    Downsample_Times = 2 ** Downsample_Level
    Downsampled_Tile_Size = Tile_Size // Downsample_Times

    Tissue_Region_Downsample_Times = 2 ** (Downsample_Level - Level)

    BOUNDS_OFFSET_PROPS = (openslide.PROPERTY_NAME_BOUNDS_X,
                           openslide.PROPERTY_NAME_BOUNDS_Y)
    BOUNDS_SIZE_PROPS = (openslide.PROPERTY_NAME_BOUNDS_WIDTH,
                         openslide.PROPERTY_NAME_BOUNDS_HEIGHT)

    Slide_Offset = tuple(int(Slide_P.properties.get(prop, 0))
                         for prop in BOUNDS_OFFSET_PROPS)
    Slide_DownSampled_Offset = [Slide_Offset[0] // Downsample_Times, Slide_Offset[1] // Downsample_Times]
    # Slide level dimensions scale factor in each axis
    Slide_Size_Scale = tuple(int(Slide_P.properties.get(prop, l0_lim))
                             for prop, l0_lim in zip(BOUNDS_SIZE_PROPS, Slide_P.dimensions))

    Slide_DownSampled_Size_Scale = [Slide_Size_Scale[0] // Downsample_Times, Slide_Size_Scale[1] // Downsample_Times]

    Downsampled_Slide_Img = Slide_P.read_region([0, 0], Downsample_Level,
                                                Slide_P.level_dimensions[Downsample_Level]).convert('L')

    Downsampled_Slide_Img = np.array(Downsampled_Slide_Img)
    Downsampled_Slide_Img = Downsampled_Slide_Img[
                            Slide_DownSampled_Offset[1]:Slide_DownSampled_Offset[1] + Slide_DownSampled_Size_Scale[1],
                            Slide_DownSampled_Offset[0]:Slide_DownSampled_Offset[0] + Slide_DownSampled_Size_Scale[0]]

    #     plt.figure(figsize=(20, 40))
    #     plt.imshow(Downsampled_Slide_Img, cmap='gray')
    #     plt.savefig('tmp_img.jpg')
    #     plt.close()

    #     # Threshold = threshold_otsu(Downsampled_Slide_Img)

    Downsampled_Slide_Binary = Downsampled_Slide_Img > 240
    #     plt.figure(figsize=(20, 40))
    #     plt.imshow(Downsampled_Slide_Binary, cmap='gray')
    #     plt.savefig('tmp_img_1.jpg')
    #     plt.close()

    i = 0
    for x in range(0, Slide_Size_Scale[0] - Tile_Size, Tile_Size - Overlap):
        for y in range(0, Slide_Size_Scale[1] - Tile_Size, Tile_Size - Overlap):
            # print(f"{Slide_Path}  Background Inference:  {i}/{len(range(0, Slide_Size_Scale[0]-Tile_Size, Tile_Size-Overlap))*len(range(0, Slide_Size_Scale[1]-Tile_Size, Tile_Size-Overlap))}")
            i = i + 1
            Downsampled_x = x // Tissue_Region_Downsample_Times
            Downsampled_y = y // Tissue_Region_Downsample_Times
            tmp_Downsampled_Slide_Binary = Downsampled_Slide_Binary[Downsampled_y:Downsampled_y + Downsampled_Tile_Size,
                                           Downsampled_x: Downsampled_x + Downsampled_Tile_Size]  # row, col -> y, x
            tmp_Downsampled_Slide_Img = Downsampled_Slide_Img[Downsampled_y:Downsampled_y + Downsampled_Tile_Size,
                                        Downsampled_x: Downsampled_x + Downsampled_Tile_Size]

            if tmp_Downsampled_Slide_Binary.size == 0:
                print('Wrong, empty region')

            # print(np.std(tmp_Downsampled_Slide_Img))

            if np.mean(tmp_Downsampled_Slide_Binary) > White_BG_Thres:
                # print(np.mean(tmp_Downsampled_Slide_Binary))
                continue  # Background
            #             elif np.count_nonzero(tmp_Downsampled_Slide_Img==0)>Downsampled_Tile_Size*Downsampled_Tile_Size*0.1:
            #                 continue # Black region

            Grid.append([x + Slide_Offset[0], y + Slide_Offset[1]])

            # print(np.std(tmp_Downsampled_Slide_Img))
            # Region_Img = Slide_P.read_region([x+Slide_Offset[0],y+Slide_Offset[1]], 0, (Tile_Size, Tile_Size)).convert('RGB')
            # Region_Img = np.array(Region_Img)
            # plt.figure()
            # plt.imshow(Region_Img)
            # plt.show()
    Slide_P.close()

    return Grid


def Stitch_Dict_Tiles(Slide_Path, Slide_Grid, Dict_Level, Dict_Tile_Size, Check_Downsample_Times):
    Slide_P = openslide.OpenSlide(Slide_Path)
    Closest_Level_in_The_Slide = Slide_P.get_best_level_for_downsample(Check_Downsample_Times + 1)
    Downsample_Times = 2 ** Closest_Level_in_The_Slide
    BOUNDS_OFFSET_PROPS = (openslide.PROPERTY_NAME_BOUNDS_X,
                           openslide.PROPERTY_NAME_BOUNDS_Y)
    BOUNDS_SIZE_PROPS = (openslide.PROPERTY_NAME_BOUNDS_WIDTH,
                         openslide.PROPERTY_NAME_BOUNDS_HEIGHT)

    Slide_Offset = tuple(int(Slide_P.properties.get(prop, 0))
                         for prop in BOUNDS_OFFSET_PROPS)
    Slide_DownSampled_Offset = [Slide_Offset[0] // Downsample_Times, Slide_Offset[1] // Downsample_Times]
    # Slide level dimensions scale factor in each axis
    Slide_Size_Scale = tuple(int(Slide_P.properties.get(prop, l0_lim))
                             for prop, l0_lim in zip(BOUNDS_SIZE_PROPS, Slide_P.dimensions))

    Slide_DownSampled_Size_Scale = [Slide_Size_Scale[0] // Downsample_Times, Slide_Size_Scale[1] // Downsample_Times]

    Slide_Img = np.ones(
        [Slide_DownSampled_Size_Scale[1] + 100, Slide_DownSampled_Size_Scale[0] + 100, 3]) * 255  # +100 for safe
    Slide_Img = Slide_Img.astype(np.uint8)
    Dict_Downsampled_Tile_Size = Dict_Tile_Size // Downsample_Times

    for i, Grid_Point in enumerate(Slide_Grid):
        print(f"Processing Grid: {i}/{len(Slide_Grid)}")
        Region_Img = Slide_P.read_region(Grid_Point, Dict_Level, (Dict_Tile_Size, Dict_Tile_Size)).convert('RGB')
        Region_Img_Resized = Region_Img.resize((Dict_Downsampled_Tile_Size, Dict_Downsampled_Tile_Size))
        Region_Img_Resized = np.array(Region_Img_Resized)
        X_Downsampled_Offset = (Grid_Point[1] - Slide_Offset[1]) // Downsample_Times
        Y_Downsampled_Offset = (Grid_Point[0] - Slide_Offset[0]) // Downsample_Times

        Slide_Img[X_Downsampled_Offset:X_Downsampled_Offset + Dict_Downsampled_Tile_Size,
        Y_Downsampled_Offset:Y_Downsampled_Offset + Dict_Downsampled_Tile_Size, :] = Region_Img_Resized

    Slide_Img_Ori_Downsampled = Slide_P.read_region(Slide_Offset, Closest_Level_in_The_Slide,
                                                    Slide_DownSampled_Size_Scale).convert('RGB')
    Slide_P.close()
    return Slide_Img, Slide_Img_Ori_Downsampled


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECDP2020 Make Tile Grid Dictionary from WSI')
    parser.add_argument('--tile_size', type=int, default=480, help='Tile size: 480 or 912')
    parser.add_argument('--test_slides_dir', type=str, default='', help='path to test slides folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Background removing model batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Background removing model cpu workers')

    args = parser.parse_args()
    Tile_Size = args.tile_size
    if Tile_Size == 480:
        Overlap = 0
    elif Tile_Size == 912:
        Overlap = 456
    else:
        print('Wrong Tile Size!!!!!!!')
        exit()

    WSI_Slides_Dir = args.test_slides_dir

    Level = 0
    White_BG_Thres = 0.75
    Check_Downsample_Times = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Tile_Dict_Save_Dir = r'tile_dict_' + str(Tile_Size)

    ## Load background removing model
    BG_Input_Size = 380
    BG_Batch_Size = args.batch_size
    BG_Num_Workers = args.num_workers
    Tissue_Prob_Thres = 0.99

    BG_Model_Path = r'checkpoints/background_remove_checkpoint.pth'
    model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 2})
    model = nn.DataParallel(model)
    model.to(device)
    ch = torch.load(BG_Model_Path)
    model.load_state_dict(ch['state_dict'])

    if not os.path.exists(Tile_Dict_Save_Dir):
        os.makedirs(Tile_Dict_Save_Dir)
    Check_Image_Save_Path = os.path.join(Tile_Dict_Save_Dir, 'Check')
    if not os.path.exists(Check_Image_Save_Path):
        os.makedirs(Check_Image_Save_Path)

    BG_Trans = transforms.Compose([
        transforms.Resize(BG_Input_Size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    Total_Dict = {
        'slides': [],
        'grid': [],
        'level': Level,
        'dict_tile_size': Tile_Size,
        'dict_overlap': Overlap
    }

    Total_Dict['slides'] = [f for f in os.listdir(WSI_Slides_Dir) if f.endswith('.mrxs')]

    for i, Slide_Name in enumerate(Total_Dict['slides']):
        # Slide_Path = r'E:\Projects\ECDP2020\Slides\TestDataset\Slides\73.mrxs'
        Slide_Path = os.path.join(WSI_Slides_Dir, Slide_Name)

        print(Slide_Path)
        Slide_Grid = Make_Grid(Slide_Path, Level, Tile_Size, Overlap, White_BG_Thres)

        Filtered_Slide_Grid = BG_Model_Filter(Slide_Grid, Slide_Path, Tile_Size, model, BG_Trans,
                                              Tissue_Prob_Thres=Tissue_Prob_Thres, BG_Batch_Size=BG_Batch_Size,
                                              BG_Num_Workers=BG_Num_Workers)
        Total_Dict['grid'].append(Filtered_Slide_Grid)

        Dict_Downsampled_Img, Slide_Img_Downsampled = Stitch_Dict_Tiles(Slide_Path, Filtered_Slide_Grid, Level,
                                                                        Tile_Size, Check_Downsample_Times)

        Slide_Img_Downsampled_Path = os.path.join(Check_Image_Save_Path, os.path.split(Slide_Path)[1][:-5] + '_Ori.jpg')
        Dict_Downsampled_Img_Path = os.path.join(Check_Image_Save_Path, os.path.split(Slide_Path)[1][:-5] + '_Dict.jpg')

        plt.figure(figsize=(20, 40))
        plt.imshow(Slide_Img_Downsampled)
        plt.savefig(Slide_Img_Downsampled_Path)
        plt.close()

        plt.figure(figsize=(20, 40))
        plt.imshow(Dict_Downsampled_Img)
        plt.savefig(Dict_Downsampled_Img_Path)
        plt.close()

    MIL_Dict_Save_Path = os.path.join(Tile_Dict_Save_Dir, 'Test_Dict.pth')
    torch.save(Total_Dict, MIL_Dict_Save_Path)
