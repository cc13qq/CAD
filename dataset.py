import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import torch
import cv2
import numpy as np
from config_cifar10 import *



def get_data_from_pt(pt_root, max_num=None, expand=None):
    # normalized data
    data_dict = torch.load(pt_root)
    adv_input_dict = data_dict['adv_inputs']
    label_dict = data_dict['labels']
    img_name_dict = data_dict['img_names']
    num_cls = len(adv_input_dict)
    
    img_list = []
    label_list = []
    img_name_list = []
    for cls in adv_input_dict:
        num = len(adv_input_dict[cls])
        if max_num is None:
            max_num = num
        img_list.append(adv_input_dict[cls][:max_num])
        label_list.append(label_dict[cls][:max_num])
        img_name_list += img_name_dict[cls][:max_num]
    adv_input_list_cat = torch.cat(img_list, 0)
    label_list_cat = torch.cat(label_list, 0)
    
    if expand is not None:
        for idx in range(len(label_list_cat)):
            y = label_list_cat[idx] + expand*num_cls
            label_list_cat[idx] = y
    
    return adv_input_list_cat, label_list_cat, img_name_list

def make_train_dict(train_dict, attack_method, _adv_train_root, sub_root=None, eps = 8, max_num=None):   
    if sub_root is None:
        sub_root = attack_method + '_eps' + str(eps) + '.pt'
    train_root = os.path.join(_adv_train_root, sub_root)
    adv_input_list_cat, label_list_cat, img_name_list =  get_data_from_pt(train_root, max_num=max_num)
    train_dict['image_list'].append(adv_input_list_cat)
    train_dict['label_list'].append(label_list_cat)
    train_dict['img_name_list'] += img_name_list
    return train_dict

def make_train_dict_expand(train_dict, attack_method, _adv_train_root, sub_root=None, eps = 8, max_num=None, expand=0):   
    if sub_root is None:
        sub_root = attack_method + '_eps' + str(eps) + '.pt'
    train_root = os.path.join(_adv_train_root, sub_root)
    adv_input_list_cat, label_list_cat, img_name_list =  get_data_from_pt(train_root, max_num=max_num, expand=expand)
    train_dict['image_list'].append(adv_input_list_cat)
    train_dict['label_list'].append(label_list_cat)
    train_dict['img_name_list'] += img_name_list
    return train_dict

def make_test_dicts(test_dicts, attack_method, _adv_test_root, sub_root=None, eps = 8, max_num=None):   
    if sub_root is None:
        sub_root = attack_method + '_eps' + str(eps) + '.pt'
    test_root = os.path.join(_adv_test_root, sub_root)
    adv_input_list_cat, label_list_cat, img_name_list =  get_data_from_pt(test_root, max_num=max_num,)
    test_dicts[attack_method] = {}
    test_dicts[attack_method]['image_list'] = adv_input_list_cat
    test_dicts[attack_method]['label_list'] = label_list_cat
    test_dicts[attack_method]['img_name_list'] = img_name_list
    return test_dicts

class torchdict_Dataset(Dataset):
    def __init__(self, adv_input_list_cat, label_list_cat, img_name_list, max_num=None):

        if max_num is not None:
            self.image_list = adv_input_list_cat[:max_num]
            self.label_list = label_list_cat[:max_num]
            self.img_name_list = img_name_list[:max_num]
        else:
            self.image_list = adv_input_list_cat
            self.label_list = label_list_cat
            self.img_name_list = img_name_list[:max_num]
        # random.shuffle(self.image_list)

    def __getitem__(self, item):
        img = self.image_list[item]
        label = self.label_list[item]
        img_name = self.img_name_list[item]
        return img, label, img_name
    
    def __len__(self):
        return len(self.image_list)
            
class torchdict2imagelist_Dataset(Dataset):
    def __init__(self, adv_input_list_cat, label_list_cat, img_name_list, transform=None, max_num=None):
        
        if max_num is not None:
            self.image_list = adv_input_list_cat[:max_num]
            self.label_list = label_list_cat[:max_num]
            self.img_name_list = img_name_list[:max_num]
        else:
            self.image_list = adv_input_list_cat
            self.label_list = label_list_cat
            self.img_name_list = img_name_list[:max_num]
        # random.shuffle(self.image_list)
        self.transform = transform
            
        self.image_list = self.image_list.permute(0, 2, 3, 1).cpu().numpy() * 255 # (b,H,W,C) numpy.array

    def __getitem__(self, item):
        img = self.image_list[item]
        img = Image.fromarray(img.astype(np.uint8)[:,:,[2,1,0]], mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.label_list[item])
        img_name = self.img_name_list[item]
        return img, label, img_name
    
    def __len__(self):
        return len(self.image_list)


    
def get_dataset_all_pt(args, attacks, w_clean = True):
    if args.dataset == 'cifar10':
        _adv_train_root = adv_train_pt_root  
        _adv_test_root = adv_test_pt_root  
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
            ])
    else:
        raise Exception

    train_dict = {}
    train_dict['image_list'] = []
    train_dict['label_list'] = []
    train_dict['img_name_list'] = []
    test_dicts = {}

    for attack in attacks:
        train_dict = make_train_dict(train_dict, attack, _adv_train_root, eps = args.eps) 
        test_dicts = make_test_dicts(test_dicts, attack, _adv_test_root, eps = args.eps)

    if w_clean:    
        attack_method = 'Clean'
        sub_root = sub_train_root
        train_dict = make_train_dict(train_dict, attack_method, cifar10_root, sub_root, eps = args.eps) 
        sub_root = sub_test_root
        test_dicts = make_test_dicts(test_dicts, attack_method, cifar10_root, sub_root, eps = args.eps)
    
    if not w_clean and attacks is None:
        train_dataset = None
    else:
        train_dict['image_list'] = torch.cat(train_dict['image_list'], 0)
        train_dict['label_list'] = torch.cat(train_dict['label_list'], 0)
        train_dataset = torchdict2imagelist_Dataset(train_dict['image_list'], train_dict['label_list'], train_dict['img_name_list'], transform=transform_train)
    
    if not w_clean and attacks is None:
        test_datasets = None
    else:
        test_datasets = {}
        for attack_method in attacks:
            test_dict = test_dicts[attack_method]
            test_datasets[attack_method] = torchdict2imagelist_Dataset(test_dict['image_list'], test_dict['label_list'], test_dict['img_name_list'], transform=transform_test)
        if w_clean:
            attack_method = 'Clean'
            test_dict = test_dicts[attack_method]
            test_datasets[attack_method] = torchdict_Dataset(test_dict['image_list'], test_dict['label_list'], test_dict['img_name_list'])

    return train_dataset, test_datasets
        
        
def get_dataset_session0_pt(args, attack, proto=False):
    train_dict = {}
    train_dict['image_list'] = []
    train_dict['label_list'] = []
    train_dict['img_name_list'] = []
    test_dicts = {}
    
    if args.dataset == 'cifar10':
        _adv_train_root = adv_train_pt_root  
        _adv_test_root = adv_test_pt_root  
        if proto:
            transform_train = transforms.Compose([
                transforms.ToTensor()
                ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
            ])
    else:
        raise Exception
        
    train_dict = make_train_dict(train_dict, attack, _adv_train_root, eps = args.eps) 
    test_dicts = make_test_dicts(test_dicts, attack, _adv_test_root, eps = args.eps)
    train_dict['image_list'] = torch.cat(train_dict['image_list'], 0)
    train_dict['label_list'] = torch.cat(train_dict['label_list'], 0)
    train_dataset_base = torchdict2imagelist_Dataset(train_dict['image_list'], train_dict['label_list'], train_dict['img_name_list'], transform=transform_train)
    test_dataset_base = torchdict2imagelist_Dataset(test_dicts[attack]['image_list'], test_dicts[attack]['label_list'], test_dicts[attack]['img_name_list'], transform=transform_test)

    return train_dataset_base, test_dataset_base


def get_dataset_fewshot_pt(args, attack, expand=0, num_shot=1, max_num=1000, proto=False):
    train_dict = {}
    train_dict['image_list'] = []
    train_dict['label_list'] = []
    train_dict['img_name_list'] = []
    test_dicts = {}
    
    if args.dataset == 'cifar10':
        _adv_train_root = adv_train_pt_root  
        _adv_test_root = adv_test_pt_root  
        if proto:
            transform_train = transforms.Compose([
                transforms.ToTensor()
                ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
            ])
    else:
        raise Exception
    
    train_dict = make_train_dict_expand(train_dict, attack, _adv_train_root, eps = args.eps, max_num=num_shot, expand=expand) 
    test_dicts = make_test_dicts(test_dicts, attack, _adv_test_root, eps = args.eps, max_num=max_num)
    train_dict['image_list'] = torch.cat(train_dict['image_list'], 0)
    train_dict['label_list'] = torch.cat(train_dict['label_list'], 0)
    train_dataset_fewshot = torchdict2imagelist_Dataset(train_dict['image_list'], train_dict['label_list'], train_dict['img_name_list'], transform=transform_train)
    test_dataset_fewshot = torchdict2imagelist_Dataset(test_dicts[attack]['image_list'], test_dicts[attack]['label_list'], test_dicts[attack]['img_name_list'], transform=transform_test)

    return train_dataset_fewshot, test_dataset_fewshot


