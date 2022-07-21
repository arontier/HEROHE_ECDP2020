import sys
import os
import numpy as np
import argparse
import random
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_curve, auc
from efficientnet_pytorch import EfficientNet
from radam import RAdam
from Image_Augmentation import Image_Augmentation
import os
import sys
import openslide
from PIL import Image
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
import pandas as pd

parser = argparse.ArgumentParser(description='ECDP2020 Test Code')
parser.add_argument('--tile_size', type=int, default=480, help='tile size: 480 or 912')
parser.add_argument('--test_lib', type=str, default='tile_dict_480/Test_Dict.pth', help='path to test wsis tile image dictionary')
parser.add_argument('--test_slides_dir', type=str, default='', help='path to test slides folder')
parser.add_argument('--checkpoint', type=str, default='checkpoints/480_CV1_checkpoint.pth', help='path of checkpoint')
parser.add_argument('--test_probs_features_lib', type=str, default='', help='path to probs and features dictionary of tile images')
parser.add_argument('--output', type=str, default='results', help='path of output result folder')
parser.add_argument('--batch_size', type=int, default=32, help='score model batch size')
parser.add_argument('--num_workers', type=int, default=8, help='score model cpu workers')

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.feature_extraction_batch_size = args.batch_size
args.workers = args.num_workers
####################################
## Do not change these parameters !!!!!!!!!!
args.wsi_batch_size = 1
args.min_max_flag = False
args.rnn_type = 'LSTM'  # RNN, LSTM, GRU
args.rnn_relu_tanh = 'relu'
args.rnn_layers = 2
args.ndims = 512
args.rnn_dropout = 0.5
args.rnn_bidirectional = False
######################################
if args.tile_size == 480:
    args.input_tile_size = 240
    args.rnn_input_size = 1280 #b0->1280 #b1->1280 b2->1408 b3->1536 b4->1792 b5->2048
    args.model_name = 'efficientnet-b1'
elif args.tile_size == 912:
    args.input_tile_size = 456
    args.rnn_input_size = 2048 #b0->1280 #b1->1280 b2->1408 b3->1536 b4->1792 b5->2048
    args.model_name = 'efficientnet-b5'
else:
    print('Wrong tile size!!!!!!!!!!!!')
    exit()

args.rnn_model_path = args.checkpoint
if not os.path.exists(args.output):
    os.makedirs(args.output)

args.csv_result_output = os.path.join(args.output, os.path.split(args.checkpoint)[-1][:-4]+'.csv')
args.template_result_path = 'CSVTEMPLATE.csv'


def main():
    # global args
    print(args)

    # score model
    score_model = EfficientNet.from_name(args.model_name, override_params={'num_classes': 2})
    score_model = nn.DataParallel(score_model)
    score_model.to(args.device)

    rnn_model = rnn_pytorch()
    rnn_model = nn.DataParallel(rnn_model)
    rnn_model.to(args.device)
    if args.rnn_model_path != '':
        model_checkpoint = torch.load(args.rnn_model_path)
        score_model.load_state_dict(model_checkpoint['score_state_dict'])
        score_model.to(args.device)
        rnn_model.load_state_dict(model_checkpoint['rnn_state_dict'])
        rnn_model.to(args.device)

    score_model.eval()
    rnn_model.eval()

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    feature_inference_trans = transforms.Compose([transforms.Resize(args.input_tile_size), transforms.ToTensor(), normalize])

    ### test lib load
    test_dset = sequence_data(args.test_lib, score_model, inference_transform=feature_inference_trans)
    feature_extraction_val_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.feature_extraction_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    test_dset.setmode(2)


    if os.path.isfile(args.test_probs_features_lib):
        test_dset.probs = torch.load(args.test_probs_features_lib)['probs']
        test_dset.features = torch.load(args.test_probs_features_lib)['features']
    else:
        test_dset.inference_grid(feature_extraction_val_loader, batch_size=args.feature_extraction_batch_size)
        probs_features_obj = {'probs': test_dset.probs,
                     'features': test_dset.features}
        test_probs_features_lib_save_path = os.path.join(args.output, os.path.split(args.checkpoint)[-1][:-4]+'_probs_features.pth')
        torch.save(probs_features_obj, test_probs_features_lib_save_path)

    slide_probs = test_single(test_dset, rnn_model)

    df_template = make_csv_results(test_dset, slide_probs, model_checkpoint, args.template_result_path)
    df_template.to_csv(args.csv_result_output, index=False)

def make_csv_results(test_dset, slide_probs, model_checkpoint, template_result_path):
    fpr = model_checkpoint['fpr']
    tpr = model_checkpoint['tpr']
    thres = model_checkpoint['thres']
    selected_thres = thres[np.argmax(tpr - fpr)]

    slide_names = [x[:-5] for x in test_dset.slidenames]
    df_template = pd.read_csv(template_result_path)
    soft_prediction = []
    hard_prediction = []
    for caseID in df_template['caseID']:
        idx = slide_names.index(str(caseID))
        soft_prediction.append(slide_probs[idx])
        if slide_probs[idx]>selected_thres:
            hard_prediction.append(1)
        else:
            hard_prediction.append(0)
    df_template['soft_prediction'] = soft_prediction
    df_template['hard_prediction'] = hard_prediction

    return df_template

def test_single(test_dset, rnn_model):
    slide_probs = []
    slide_num = len(test_dset.slidenames)
    slide_idxs = [i for i in range(slide_num)]
    slide_idxs_batch_sampled = [slide_idxs[i:i + args.wsi_batch_size] for i in
                                range(0, len(slide_idxs), args.wsi_batch_size)]
    for slide_batch_i, slide_idxs_batch in enumerate(slide_idxs_batch_sampled):
        print(f'[{slide_batch_i}/{slide_num}] Processing {test_dset.slidenames[slide_batch_i]}')
        output_batch = []
        for the_slide_idx in slide_idxs_batch:
            the_slide_features = test_dset.make_the_slide_tile_data(the_slide_idx)
            the_slide_features = the_slide_features.to(args.device)
            the_slide_features = the_slide_features.unsqueeze(1)
            output = rnn_model(the_slide_features)
            output_batch.append(output.squeeze(0))

        output_batch = torch.stack(output_batch)
        batch_probs = F.softmax(output_batch, dim=1)
        slide_probs.extend(batch_probs.detach()[:, 1].clone().cpu().numpy().tolist())

    return slide_probs



class rnn_pytorch(nn.Module):

    def __init__(self):
        super(rnn_pytorch, self).__init__()
        if args.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=args.rnn_input_size, hidden_size=args.ndims, num_layers=args.rnn_layers,
                         nonlinearity=args.rnn_relu_tanh, dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)
        elif args.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=args.rnn_input_size, hidden_size=args.ndims, num_layers=args.rnn_layers,
                                 dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)
        elif args.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=args.rnn_input_size, hidden_size=args.ndims, num_layers=args.rnn_layers,
                               dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)
        if args.rnn_bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.fc = nn.Linear(args.ndims*self.num_directions, 2)

    def forward(self, input):

        output, h_n = self.rnn(input)
        output = self.fc(output[-1])
        return output


class sequence_data(data.Dataset):

    def __init__(self, path, score_model, inference_transform=None):

        lib = torch.load(path)
        self.inference_transform = inference_transform
        self.mode = None
        self.slidenames = lib['slides']
        self.grid = []
        self.slideIDX = []
        ### for test
        # for i, g in enumerate(lib['grid']):
        #     self.grid.extend(g[:30])
        #     self.slideIDX.extend([i] * len(g[:30]))

        for i, g in enumerate(lib['grid']):
            self.grid.extend(g)
            self.slideIDX.extend([i] * len(g))

        self.level = lib['level']
        self.dict_tile_size = lib['dict_tile_size']
        self.dict_overlap = lib['dict_overlap']
        print('Dict Tile Size: {}'.format(self.dict_tile_size))
        print('Dict Overlap: {}'.format(self.dict_overlap))

        slides = []
        for i, slidename in enumerate(self.slidenames):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(self.slidenames)))
            sys.stdout.flush()
            slide_path = os.path.join(args.test_slides_dir,slidename)
            slides.append(openslide.OpenSlide(slide_path))

        print('')
        self.slides = slides
        self.score_model = score_model
        self.probs = None
        self.features = None

    def setmode(self, mode):
        self.mode = mode

    def inference_grid(self, loader, batch_size=16):
        probs = torch.FloatTensor(len(loader.dataset))
        features = torch.FloatTensor(len(loader.dataset), args.rnn_input_size)
        with torch.no_grad():
            for i, input in enumerate(loader):
                print('Tile Probs Inference \tBatch: [{}/{}]'.format(i + 1, len(loader)))
                input = input.to(args.device)
                feature = self.score_model.module.extract_features(input).to(args.device)
                feature = F.adaptive_avg_pool2d(feature, 1)
                feature = feature.view(input.size(0), -1)
                features[i * batch_size:i * batch_size + input.size(0),:] = feature.detach().clone()

                output = F.softmax(self.score_model(input), dim=1)
                probs[i * batch_size:i * batch_size + input.size(0)] = output.detach()[:, 1].clone()
        self.probs = probs.cpu().numpy()
        self.features = features
        # self.probs = np.random.rand(len(loader.dataset)).astype(np.float32)

    def make_the_slide_tile_data(self, selected_slide_idx):
        slideIDX = np.array(self.slideIDX)
        probs = np.array(self.probs)
        self.the_slide_grid_idxs = np.where(slideIDX == selected_slide_idx)[0]
        self.the_slide_probs = probs[self.the_slide_grid_idxs]
        self.the_slide_features = self.features[self.the_slide_grid_idxs,:]
        if args.min_max_flag:
            the_slide_probs_idxs_ordered = np.argsort(self.the_slide_probs)
            probs_idxs_selected = the_slide_probs_idxs_ordered[:len(the_slide_probs_idxs_ordered)//20].tolist()\
                                 +the_slide_probs_idxs_ordered[-len(the_slide_probs_idxs_ordered)//20:].tolist()
            return self.the_slide_features[probs_idxs_selected]
        else:
            return self.the_slide_features


    def __getitem__(self, index):
        if self.mode == 1:
            out = []
            slideIDX, coord, target = self.t_data[index]

            for i in range(self.s):
                img = self.slides[slideIDX[i]].read_region(coord[i], self.level,
                                                           (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

                if self.inference_transform is not None:
                    img = self.inference_transform(img)
                out.append(img)
            return out, target[0]

        elif self.mode == 2:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord, self.level,
                                                    (self.dict_tile_size, self.dict_tile_size)).convert('RGB')

            if self.inference_transform is not None:
                img = self.inference_transform(img)
            return img

    def __len__(self):
        if self.mode == 1:
            return len(self.t_data)

        elif self.mode == 2:
            return len(self.grid)


if __name__ == '__main__':
    main()
