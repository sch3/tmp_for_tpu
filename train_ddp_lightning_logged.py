import os
import sys
from datetime import datetime
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision
import glob
from PIL import Image
from google.cloud import storage
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
import io
import time
from torch.optim.lr_scheduler import _LRScheduler
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint,Callback
from torch.optim.lr_scheduler import _LRScheduler
from p_tqdm import p_map
storage_client = storage.Client()
bucket = storage_client.bucket('kneron-eval-training-bucket')
#import torchmetrics
#—net r50 —block-size ‘32 32 64 128 256’ —se 0 —focal True —emb 256 —do 0.0 —margin 0.5 —gr
def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument('--inp-size', type=int, default=112, help='input image size')
    parser.add_argument('--num_workers', type=int, default=10, help='num workers')
    parser.add_argument('--block-layout', type=str, default='8 28 6', help='feature block layout')
    #parser.add_argument('--block-size', type=str, default='32 384 1152 2144', help='feature block size')
    parser.add_argument('--block-size', type=str, default='32 32 64 128 256', help='feature block size') # original size
    #parser.add_argument('--block-size', type=str, default='48 48 72 160 288', help='feature block size')
    parser.add_argument('--se-ratio', type=int, default=0, help='SE reduction ratio')
    parser.add_argument('--head', type=str, default='fc', help='head fc or varg')
    #parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--emb_size','--embsize','--emb-size', type=int, default=256, help='embedding length')
    #parser.add_argument('--do-rate', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--do-rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    #parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
    parser.add_argument('--focal-loss', type=bool, default=True, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--checkpoint', type=str, default='/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/model/checkpoint/kface_fr-r50_kl520-I112-E256-e0194-av0.9978_0.9785_0.9769_0.9970_0.9533_0.9451.pth', help='checkpoint')


    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--warmup', type=int, default=0, help='warmup training epochs without validation')
    parser.add_argument('--cooldown', type=int, default=0, help='keep training with repeating the last few epochs')
    parser.add_argument('--max-cool', type=int, default=5, help='Maxium cooling down without improvment')
    parser.add_argument('--end-epoch', type=int, default=100, help='training epoch size.')
    parser.add_argument('--gpus', type=str, default='0', help='running on GPUs ID')
    parser.add_argument('--num_gpus', type=int, default=0, help='num lightning gpus')
    #parser.add_argument('--log-dir', type=str, default=None, help='Checkpoint/log root directory')
    parser.add_argument('--log-dir', type=str, default='/home/steven_ho/model_training/logs', help='Checkpoint/log root directory')
    #parser.add_argument('-gr', '--grayscale', help='Use grayscale input.', action='store_true')
    parser.add_argument('-nf', '--no-flip', help='No face flip in evaluation.', action='store_false')
    parser.add_argument('-pre', '--pre-norm', help='Preprocessing normalization id.', default='CV-kneron')
    parser.add_argument('--finetune', help='whether finetune from some checkpoint .', action='store_true')
    parser.add_argument('--large_resnet', help='large_resnet.', action='store_true')
    parser.add_argument('--regnet', help='regnet600MF.', action='store_true')
    parser.add_argument('--dbface', help='dbface_mbv2.', action='store_true')
    parser.add_argument('--curricular', help='Curricular Loss.', action='store_true')

    parser.add_argument('--resnest', help='resnest.', action='store_true')
    parser.add_argument('--dropblock', help='dropblock.', action='store_true')
    parser.add_argument('--resnet50_bottleneck', help='resnet50_bottleneck.', action='store_true')
    parser.add_argument('--regnet800', help='regnet800MF.', action='store_true')
    parser.add_argument('--new_metric_fc', help='whether adjust the total label counts', action='store_true')
    parser.add_argument('--freeze_backbone', help='set model to eval()', action='store_true')
    parser.add_argument('--partial_freeze', help='partial_freeze', action='store_true')
    parser.add_argument('--upsample', help='upsample', action='store_true')
    parser.add_argument('--upsample_child', help='upsample_child', action='store_true')

    parser.add_argument('--glass', help='w/ glass.', action='store_true')
    parser.add_argument('--lighting', help='w/ lighting. (with glasses)', action='store_true')
    parser.add_argument('--mi_augment', help='w/ mi_augment (2020/12/17) from Andy Wei', action='store_true')

    parser.add_argument('--occlusion_ratio', type=float, default=1.0, help='--occlusion_ratio')
    parser.add_argument('--attribute_filter', help='attribute_filter', action='store_true')
    parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0314_10_label_v23/checkpoint-r50-I112-E512-e0099-av0.9987_0.9990.tar', help='nir_head')
    parser.add_argument('--max_epoch', type=int, default=10, help='--occlusion_ratio')

    args = parser.parse_args()
    args.block_layout = [int(n) for n in args.block_layout.split()]
    args.block_size = [int(n) for n in args.block_size.split()]
    args.gpus = [int(n) for n in args.gpus.split()]

    return args


args = parse_args()
'''
Loss and Accuracy
'''
class FocalLoss(nn.Module):

    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
def load_image_local(im):
    img = Image.open(im)
    return img.convert('RGB')
'''
Dataset and transforms
'''
def load_image(im):
    blob = bucket.blob(im)
    img = Image.open(io.BytesIO(blob.download_as_string()))
    return img.convert('RGB')

data_transforms = {
    'kneron': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[1.0, 1.0, 1.0], mean=[0.5, 0.5, 0.5])
    ]),
    'tf': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
    ]),
    'lfw': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[128., 128., 128.], mean=[128., 128., 128.])
    ]),
    'darker': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[1., 1., 1.], mean=[0., 0., 0.])
    ]),
    'CV-kneron': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[256., 256., 256.], mean=[128, 128, 128]),
    ]),
}


class ImageDataSet(Dataset):
    image_list = []
    label_list = []

    def __init__(self, image_root=None, image_list=None, image_transform_indoor=None, image_transform_outdoor=None, image_transform= None, data_transform=None, min_images_per_label=10, with_glass = False, with_lighting = False, with_mi_augment = False, upsample = False, upsample_child = False, occlusion_ratio=1.0, attribute_filter=False):
        self.image_transform_indoor = image_transform_indoor
        self.image_transform_outdoor = image_transform_outdoor
        self.image_transform = image_transform
        self.data_transform = data_transform
        self.min_images_per_label = min_images_per_label
        self.image_root = image_root

        self.image_list = image_list
        self.with_glass = with_glass
        self.with_lighting = with_lighting
        self.with_mi_augment = with_mi_augment
        self.upsample = upsample
        self.upsample_child = upsample_child
        self.occlusion_ratio = occlusion_ratio
        self.attribute_filter = attribute_filter


        # Open and load text file including the whole training data
        if self.image_list == None:
            # load from image_root
            print('Loading data from directory:\n', self.image_root)
            self.image_list, self.label_list, self.images_per_label, self.upsample_index, self.upsample_index_child = self._get_dataset_from_dir(self.image_root, self.min_images_per_label)
        else:
            if self.image_root == None:
                self.image_root = os.path.dirname(self.image_list)
            print('Loading data from list:', self.image_list)

            self.image_list, self.label_list, self.images_per_label = self._get_dataset_by_csv(
                    self.image_list, self.image_root, min_images_per_label=self.min_images_per_label)

        #print ("self.label_list: ", self.label_list)
        #exit(0)
        self.min_num_images = min(self.images_per_label.values())
        self.max_num_images = max(self.images_per_label.values())
        self.num_labels = len(self.images_per_label)
        print('- number of labels:', self.num_labels)
        print('- total number of images:', self.__len__())
        print('- number of images per label, min/max: {}/{}'.format(self.min_num_images, self.max_num_images))
        #preload:        
        #self.image_list_local =p_map(load_image,self.image_list)
        
    def __getitem__(self, index):
        #img = self.image_list_local[index]
        img = load_image(self.image_list[index])
        if 'indoor' in self.image_list[index]:
            img = self.image_transform_indoor(img)
        elif 'outdoor' in self.image_list[index]:
            img = self.image_transform_outdoor(img)
        else:
            #raise NotImplementedError
            img = self.image_transform(img)
        img = np.asarray(img, dtype='f')
        if self.data_transform:
            img = self.data_transform(img)
        label = self.label_list[index]
        return img, label


    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.image_list)


    def _get_dataset_from_dir(self,img_root_list, min_images_per_label=10):
        image_list = []
        label_list = []
        images_per_label = {}
        upsample_index = {}
        upsample_index_child = {}
        label_seq = 0
        missglass_counter = 0
        for img_root in img_root_list:
            print ("=============================================")
            print("img_root: ", img_root)
            print ("without glasses")
            path_exp = os.path.expanduser(img_root)
            files =  storage_client.list_blobs('kneron-eval-training-bucket', prefix=img_root)
            labels = []
            ims_per_label={}
            for blob in tqdm(files):
                if '.txt' not in blob.name:
                    l = blob.name.split('/')[-2]
                    if l not in labels:
                        labels.append(l)
                        ims_per_label[l]=[blob.name]
                    else:
                        ims_per_label[l].append(blob.name)
            labels = sorted(labels)
            labels = [item for item in labels if '.txt' not in item]
            # may sorted as str
            labels = np.asarray(labels, dtype=str)
            #labels = np.asarray(labels, dtype=int)
            labels = np.sort(labels)
            print("len(labels):",  len(labels))
            for label in tqdm(labels):
                image_paths =  ims_per_label[label]
                images_in_label = len(image_paths)
                if 'mi8'in img_root and self.with_mi_augment:
                    facedir_mi_augment= os.path.join(path_exp_mi_augment, str(label))
                    image_paths_mi_augment = glob.glob(os.path.join(facedir_mi_augment, '*'))
                    image_paths.extend(image_paths_mi_augment)

                if images_in_label >= min_images_per_label:
                    images_per_label.update({label_seq:images_in_label})
                    if self.upsample or self.upsample_child:
                        if 'mi8'in img_root:
                            upsample_index.update({label_seq:True})
                            if self.upsample_child:
                                image_name = os.path.basename(image_paths[0])
                                split_image_name = image_name.split('_') #age
                                if int(split_image_name[3]) <16:
                                    upsample_index_child.update({label_seq:True})
                                else:
                                    upsample_index_child.update({label_seq:False})
                        else:
                            upsample_index.update({label_seq:False})
                            upsample_index_child.update({label_seq:False})
                    for path in image_paths:
                        image_list.append(path)
                        label_list.append(label_seq)
                    label_seq += 1
        return image_list, label_list, images_per_label, upsample_index, upsample_index_child
image_transforms = {
    'kneron-gray': torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01), #origin
    ]),
    'kneron-gray-indoor': torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
    ]),

    'kneron-gray-outdoor': torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),

        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
    ]),
}


'''
Net
'''

DEFAULT_LAYER = 50
BLOCK_LAYOUT = {
    18:  [2, 2,  2, 2],
    34:  [3, 4,  6, 3],
    34_2:  [3, 4,  10, 3],
    34_3:  [3, 4,  12, 3],
    #34_3:  [3, 4,  10, 3],
    50:  [3, 4, 14, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
    }

DEFAULT_SIZE = 'm'
BLOCK_SIZE = {
    's' : [16, 16, 32, 64, 128],
    'se': [32, 32, 64, 64, 128],
    'sf': [32, 32, 64, 128, 128],
    'm' : [32, 32, 64, 128, 256],
    'ml' : [64, 64, 96, 128, 256],
    'l' : [64, 64, 128, 256, 512],
    'l1': [64, 64, 128, 512, 512],
    'l2': [64, 64, 256, 256, 512],
    'l3': [64, 64, 256, 512, 512],
    'l7': [64, 128, 256, 512, 512],
    'x_8': [64, 128, 256, 512, 1024],
    'x'  : [128, 128, 256, 512, 1024],
    'large' : [48, 48, 72, 160, 288],
    }
def head_block(inplanes, feat_size, do_rate, emb_size, head='fc', varg_last=1024):
        # use grouped head, otherwise go fc
        if head=='varg':
            return nn.Sequential(
                    nn.Conv2d(inplanes, varg_last, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(varg_last),
                    nn.PReLU(varg_last),
                    nn.Conv2d(varg_last, varg_last, kernel_size=feat_size, stride=1, padding=0, bias=False, groups=varg_last//8),
                    nn.BatchNorm2d(varg_last),
                    nn.Conv2d(varg_last, varg_last//2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(varg_last//2),
                    nn.PReLU(varg_last//2),
                    nn.Flatten(),
                    nn.Linear(varg_last//2, emb_size),
                    nn.BatchNorm1d(emb_size)
                    )
        else:
            return nn.Sequential(
                    #nn.BatchNorm2d(inplanes),
                    nn.Dropout(p=do_rate),
                    nn.Flatten(),
                    nn.Linear(inplanes*feat_size*feat_size, emb_size),
                    nn.BatchNorm1d(emb_size)
                    )
def input_block(inp_size, inplanes):
    if inp_size == 112:
        return nn.Sequential(
                nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.PReLU(inplanes),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
    elif inp_size == 224:
        return nn.Sequential(
                nn.Conv2d(3, inplanes//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(inplanes//2),
                nn.PReLU(inplanes//2),
                nn.Conv2d(inplanes//2, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.PReLU(inplanes),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, se_ratio=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se_ratio=0):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, affine=False)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU(inplanes)
        self.prelu_out = nn.PReLU(planes)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if se_ratio > 0:
            self.se = SEBlock(planes, se_ratio)
        else:
            self.se = None
        #self.bnbp = nn.BatchNorm2d(planes, affine=False)


    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        #out = self.bnbp(out)
        out = self.prelu_out(out)

        return out
'''
Lightning Model
'''
#lightning model 
#checkpoint saves under the hood in lightning logs by default. will also save last checkpoint for easy resume if specified in callback
metrics = {"backbone inference":[],"head inference":[],"criterion":[],"accuracy":[],"total train step":[],"val backbone inference":[],"val head inference":[],"val accuracy":[],"total val step":[]}
class LitModel(pl.LightningModule):
    def __init__(self,num_labels):
        super().__init__()
        #resnet 18
        self.expend_ratio = 4
        block_size = args.block_size
        layout = BLOCK_LAYOUT[18]
        se_ratio = args.se_ratio
        inp_size = args.inp_size
        block = IRBlock
        self.input = input_block(inp_size, block_size[0])
        self.inplanes = block_size[0]
        self.layer1 = self._make_layer(block, block_size[0], block_size[1], layout[0], stride=1, se_ratio=se_ratio)
        self.layer2 = self._make_layer(block, block_size[1], block_size[2], layout[1], stride=2, se_ratio=se_ratio)
        self.layer3 = self._make_layer(block, block_size[2], block_size[3], layout[2], stride=2, se_ratio=se_ratio)

        self.layer4 = self._make_layer(block, block_size[3], block_size[4], layout[3], stride=2, se_ratio=se_ratio)
        self.head = head_block(block_size[4], 7, args.do_rate, args.emb_size, args.head)
        #arc margin
        self.num_classes = num_labels
        self.weight = Parameter(torch.FloatTensor(self.num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
        #other
        self.criterion =  FocalLoss(gamma=args.gamma)#nn.CrossEntropyLoss()
    def arc_margin_rgb(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        #print("cosine.shape: ", cosine.shape) #cosine.shape:  torch.Size([8, 280])  (Batcg x class)
        #exit(0)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, se_ratio=0):
        downsample = None

        if block == Bottleneck:
            if stride != 1 or self.inplanes != (self.expend_ratio*planes):
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes*self.expend_ratio, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes*self.expend_ratio),
                )
        else:
            if stride != 1 or inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        if block == Bottleneck:
            layers = []
            layers.append(block(self.inplanes,  planes, stride, downsample, se_ratio=se_ratio))
            self.inplanes = planes*self.expend_ratio
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, se_ratio=se_ratio))
        else:
            layers = []
            layers.append(block(inplanes,  planes, stride, downsample, se_ratio=se_ratio))
            for i in range(1, blocks):
                layers.append(block(planes, planes, se_ratio=se_ratio))

        return nn.Sequential(*layers)
    
    #for inference/prediction, can be invoked by self(x). separate from train step
    def forward(self, x):
        #with torch.cuda.amp.autocast(enabled=False):
            x = self.input(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.head(x)
            return x
    #main train loop, is separate from forward
    #other ops under hood: set model train, clear gradients,update parameters
    def training_step(self, batch, batch_idx):
        start_total = time.time()
        img,label = batch
        start = time.time()
        feature = self(img)
        end = time.time()
        metrics["backbone inference"].append(end - start)
        start = time.time()
        output=self.arc_margin_rgb(feature,label)
        end = time.time()
        metrics["head inference"].append(end - start)
        start = time.time()
        loss = self.criterion(output, label)
        end = time.time()
        metrics["criterion"].append(end - start)
        start = time.time()
        top5_accuracy = accuracy(output, label, 5)
        end = time.time()
        metrics["accuracy"].append(end - start)
        self.log("train_acc5",top5_accuracy)
        end_total = time.time()
        metrics["total train step"].append(end_total - start_total)
        return loss
    #validation step: will call model train step. WIll set eval step, evaluate, revert to train mode
    def validation_step(self,batch,batch_idx):
        start_total = time.time()
        img,label = batch
        start = time.time()
        feature = self(img)
        end = time.time()
        metrics["val backbone inference"].append(end - start)
        start = time.time()
        output =self.arc_margin_rgb(feature,label)
        end = time.time()
        metrics["val head inference"].append(end - start)
        start = time.time()
        top5_accuracy = accuracy(output, label, 5)
        end = time.time()
        metrics["val accuracy"].append(end - start)
        self.log("val_loss",self.criterion(output, label))
        end_total = time.time()
        metrics["total val step"].append(end_total - start_total)
        return top5_accuracy
    #for use with accelerator across batch. should sum together validation batch outputs. 
    def validation_epoch_end(self, validation_step_outputs):
        top5_accuracy = sum(validation_step_outputs) / len(validation_step_outputs)
        self.log("val_top5_acc",top5_accuracy,on_step=False, on_epoch=True)
        return top5_accuracy
    def configure_optimizers(self):
        optimizer = torch.optim.SGD([{'params': self.parameters()}],
            lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch, last_epoch=-1)
        return {"optimizer":optimizer,"lr_scheduler":scheduler}
    #training step end only necessary if using dp or dpp2 (ie multi node ddp) for aggregating batch results on single 
    #test step loop
    def test_step(self,batch,batch_idx):
        img,label = batch
        output = self(img)
        output = self.arc_margin_rgb(output,label)
        top1_accuracy = accuracy(output, label, 1)
        return top1_accuracy
    #for use with accelerator across batch. should sum together validation batch outputs. 
    def test_epoch_end(self, output_results):
        top1_accuracy = sum(output_results) / len(output_results)
        self.log("top1_accuracy",top1_accuracy)
        return top1_accuracy
#val and train datasets
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset=None, image_root=None, image_list=None,
            image_transform_indoor=None, image_transform_outdoor=None, image_transform=None, data_transform=None,
            shuffle=True, images_per_label=10, batch_size=64, num_workers=10, glass = False, lighting = False, mi_augment = False, upsample = False, upsample_child = False, occlusion_ratio=1.0, attribute_filter=False, **kwargs):
        super().__init__()
        self.dataset =ImageDataSet(image_root=image_root, image_list=image_list,
                image_transform_indoor=image_transform_indoor, image_transform_outdoor=image_transform_outdoor, image_transform=image_transform, data_transform=data_transform,
                min_images_per_label=images_per_label, with_glass = glass, with_lighting = lighting, with_mi_augment = mi_augment, upsample = upsample, upsample_child = upsample_child, occlusion_ratio=occlusion_ratio, attribute_filter=attribute_filter)
        self.batch_size= batch_size
        train_size = int(len(self.dataset)*0.75)
        self.tr, self.tst = torch.utils.data.random_split(self.dataset, [train_size,len(self.dataset)-train_size])
        self.dataloader = DataLoader(self.tr.dataset,batch_size=self.batch_size,num_workers=num_workers, pin_memory=True)
    def train_dataloader(self):
        return self.dataloader
    def val_dataloader(self):
        return  self.dataloader
    def test_dataloader(self):
        return DataLoader(self.tst.dataset,batch_size=self.batch_size,num_workers=10, pin_memory=True)

#test dataloader
class TestImageDataModule(pl.LightningDataModule):
    def __init__(self,val_folder, same_pairs_path, diff_pairs_path, to_grayscale=False):
        super().__init__()
        self.dataset = ValImageDataset(val_folder, same_pairs_path, diff_pairs_path, to_grayscale=to_grayscale)
    def test_dataloader(self):
        return DataLoader(self.dataset,num_workers=10, pin_memory=True)
    
if __name__ == '__main__':
    print("Visible devices ",torch.cuda.device_count(),' ',torch.cuda.is_available())
    #set fixed seed
    pl.seed_everything(7)
    #data loader
    DATA_TRANSFORM_ID = 'CV-kneron' #'kneron'
    IMG_TRANSFORM_ID_INDOOR = 'kneron-gray-indoor'
    IMG_TRANSFORM_ID_OUTDOOR = 'kneron-gray-outdoor'
    IMG_TRANSFORM_ID = 'kneron-gray'
    #create dataset
    #
    IMAGE_ROOT =['ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign_truncated']#['toy_bucket']#
    IMAGE_LIST=None
    NUM_WORKERS=args.num_workers
    IMAGE_PER_LABEL = 10
    train_loader = ImageDataModule(image_root=IMAGE_ROOT, image_list=IMAGE_LIST,
        image_transform_indoor=image_transforms[IMG_TRANSFORM_ID], image_transform_outdoor=image_transforms[IMG_TRANSFORM_ID],
        image_transform=image_transforms[IMG_TRANSFORM_ID],
        data_transform=data_transforms[DATA_TRANSFORM_ID],
        batch_size=args.batch_size, images_per_label=IMAGE_PER_LABEL, shuffle=True, num_workers=NUM_WORKERS, glass = args.glass, lighting = args.lighting,
        mi_augment = args.mi_augment, upsample = args.upsample, upsample_child = args.upsample_child, occlusion_ratio=args.occlusion_ratio, attribute_filter=args.attribute_filter)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_top5_acc",
        dirpath="lightning_logs",
        filename="sample-resnet18-{epoch:02d}",
        save_top_k=3, #save top k 
        mode="max",
        save_last=True
    )
    #seems to save ckpt with check val epoch frequency and is defined here instead of ckpt callback
    #https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    #TPU settings
    trainer = pl.Trainer(tpu_cores=8, max_epochs=args.max_epoch,callbacks=[checkpoint_callback],profiler="simple",check_val_every_n_epoch=10)
    #trainer = pl.Trainer(gpus=args.num_gpus,accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False),max_epochs=args.max_epoch,callbacks=[checkpoint_callback],profiler="simple")
    print("NUM LABELS ",train_loader.dataset.num_labels)
    model = LitModel(train_loader.dataset.num_labels)
    
    trainer.fit(model, train_loader)
    for f in os.listdir("/home/steven_ho/lightning_logs"):
        try:
            if ".ckpt" in f and "tmp" not in f: 
                print("Checkpoint ",f)
                trainer.test(ckpt_path=os.path.join("/home/steven_ho/lightning_logs",f),dataloaders=train_loader)
        except:
            continue
