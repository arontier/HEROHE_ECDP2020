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
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECDP2020 CSV Results Ensemble')
    parser.add_argument('--results_dir', type=str, default='results', help='path of output results directory includes csv files')
    parser.add_argument('--hard_pred_mode', type=int, default=0, help='hard prediction mode: 0-> Max Voting, 1->averaged soft prediction thresholding ')

    template_result_path = 'CSVTEMPLATE.csv'

    args = parser.parse_args()

    csv_dir = args.results_dir
    mode = args.hard_pred_mode
    csv_paths = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

    df_template = pd.read_csv(template_result_path)

    soft_prediction = []
    hard_prediction = []
    for csv_path in csv_paths:
        df_tmp_result = pd.read_csv(csv_path)
        soft_prediction.append(df_tmp_result['soft_prediction'].tolist())
        hard_prediction.append(df_tmp_result['hard_prediction'].tolist())

    soft_prediction_template = np.mean(soft_prediction, axis=0)
    hard_prediction_sum = np.sum(hard_prediction, axis=0)
    print(len(np.where(hard_prediction_sum==5)[0]))
    print('probs of hard prediction sum = 5')
    hard_prediction_template = []
    if mode == 0:
        for i, tmp_hard_prediction in enumerate(hard_prediction_sum):
            if tmp_hard_prediction > 5:
                hard_prediction_template.append(1)
            elif tmp_hard_prediction < 5:
                hard_prediction_template.append(0)
            elif tmp_hard_prediction == 5:
                print(soft_prediction_template[i])
                if soft_prediction_template[i] > 0.5:
                    hard_prediction_template.append(1)
                else:
                    hard_prediction_template.append(0)
    else:
        for i, tmp_soft_prediction_template in enumerate(soft_prediction_template):
            if tmp_soft_prediction_template > 0.5:
                hard_prediction_template.append(1)
            else:
                hard_prediction_template.append(0)

    df_template['soft_prediction'] = soft_prediction_template
    df_template['hard_prediction'] = hard_prediction_template

    df_template.to_csv('Arontier_HYY.csv', index=False)
