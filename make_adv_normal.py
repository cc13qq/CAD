import torch
import os
from PIL import Image
import torchvision as tv
import torchattacks
from torchvision import transforms
from utils import *
import time
import argparse
from networks import WideResNet
from torch.utils.data import Dataset
from config_cifar10 import *


def get_atk_method(atk_method, model, params, normalize = None, trainloader=None):
    '''
    eps = 8/255
    '''
    if 'eps' in params: eps = params['eps'] / 255
    if 'alpha' in params: alpha = params['alpha'] / 255
    if 'steps' in params: steps = params['steps']
    if 'std' in params: std = params['std']

    if atk_method == 'FGSM':
        atk = torchattacks.FGSM(model, eps=eps)
    elif atk_method == 'BIM':
        atk = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'RFGSM':
        atk = torchattacks.RFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'PGD':
        atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'MIFGSM':
        atk = torchattacks.MIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'DIFGSM':
        atk = torchattacks.DIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'TIFGSM':
        atk = torchattacks.TIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    
    # new version
    elif atk_method == 'NIFGSM':
        atk = torchattacks.NIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'SINIFGSM':
        atk = torchattacks.SINIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'VNIFGSM':
        atk = torchattacks.VNIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'VMIFGSM':
        atk = torchattacks.VMIFGSM(model, eps=eps, alpha=alpha, steps=steps)


    elif atk_method == 'OnePixel':
        atk = torchattacks.OnePixel(model, steps=steps)
    elif atk_method == 'SparseFool':
        atk = torchattacks.SparseFool(model, steps=steps)
    elif atk_method == 'Pixle':
        atk = torchattacks.Pixle(model)
    elif atk_method == 'JSMA':
        atk = torchattacks.JSMA(model)
    
    elif atk_method == 'FAB':
        atk = torchattacks.FAB(model, eps=eps, steps=steps)
    elif atk_method == 'LGV':
        assert trainloader is not None
        atk = torchattacks.LGV(model, trainloader=trainloader)
    elif atk_method == 'EADL1':
        atk = torchattacks.EADL1(model)
    elif atk_method == 'EADEN':
        atk = torchattacks.EADEN(model)
    
    elif atk_method == 'CW':
        atk = torchattacks.CW(model)
    elif atk_method == 'DeepFool':
        atk = torchattacks.DeepFool(model)
    elif atk_method == 'APGDL2':
        atk = torchattacks.APGD(model, eps=0.5, steps=steps, norm="L2")
    elif atk_method == 'AutoAttackL2':
        atk = torchattacks.AutoAttack(model, eps=0.5, norm="L2")
    elif atk_method == 'SquareL2':
        atk = torchattacks.Square(model, eps=0.5, norm="L2")
    elif atk_method == 'APGD':
        atk = torchattacks.APGD(model, eps=eps, steps=steps)
    elif atk_method == 'AutoAttack':
        atk = torchattacks.AutoAttack(model, eps=eps)
    elif atk_method == 'Square':
        atk = torchattacks.Square(model, eps=eps)
        
    else:
        raise Exception

    if normalize:
        atk.set_normalization_used(normalize['mean'], normalize['std'])
        atk._set_normalization_applied(False)
    return atk


CIFAR10_LABEL = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                 "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}

class ImageList_Dataset(Dataset):
    def __init__(self, data_root, transform, label, max_num=None):

        self.transform = transform

        if max_num is not None:
            self.image_list = os.listdir(data_root)[:max_num]
        else:
            self.image_list = os.listdir(data_root)

        self.label = label
        self.data_root = data_root
        # random.shuffle(self.image_list)

    def __getitem__(self, item):
        # [img_path, gt_str] = self.image_list[item].split('\t')
        img_name = self.image_list[item]
        img_path = os.path.join(self.data_root, img_name)
        img = Image.open(img_path)
        img = img.convert("RGB")
        label = self.label
        img = self.transform(img)
        return img, label, img_name
    
    def __len__(self):
        return len(self.image_list)

def run_normal_cls(atk_method, dataset_name, batch_size, base_data_root, adv_data_root, model, img_shape, params, save_img, device):

    transform_raw_to_clf = raw_to_clf()
    print('\n Attack dataset ', dataset_name)
    normalize = {}
    normalize['mean'] = (0.5, 0.5, 0.5)
    normalize['std'] = (0.5, 0.5, 0.5)

    transform = transforms.Compose([transforms.ToTensor()])
    trainroot = cifar10_root
    traindataset = tv.datasets.CIFAR10(root=trainroot, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=False)
    
    atk = get_atk_method(atk_method, model, params, normalize, trainloader)
    print(' Attack algorithm ', str(atk))

    class_names = os.listdir(base_data_root)
    num_class = len(class_names)
    att_acc_total = 0
    nat_acc_total = 0
    for cls in class_names:
        sub_root = atk_method + '/eps' + str(params['eps'])
        save_root = os.path.join(adv_data_root, sub_root, cls)
        if not os.path.exists(save_root):
            os.makedirs(save_root) # 创建数据保存路径
        else:
            return
        
        cls_root = os.path.join(base_data_root, cls)
        label = CIFAR10_LABEL[cls]
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = ImageList_Dataset(cls_root, transform, label)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_all = 0
        att_correct = 0
        nat_correct = 0
        for batch, (X_batch, y_batch, img_name) in enumerate(test_dataloader):

            num_all += X_batch.shape[0]

            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            adv_images = atk.__call__(X_batch, y_batch) # (b,C,H,W) torch.tensor

            logits_nat = model(transform_raw_to_clf(X_batch))
            nat_correct_batch = torch.eq(torch.argmax(logits_nat, dim=1), y_batch).float().to('cpu').sum()
            
            logits_att = model(transform_raw_to_clf(adv_images))
            att_correct_batch = torch.eq(torch.argmax(logits_att, dim=1), y_batch).float().to('cpu').sum()

            att_correct += att_correct_batch
            nat_correct += nat_correct_batch

            # print("batch {}: att_correct {} in {}".format(batch, att_correct_batch, X_batch.shape[0]))

            adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() * 255 # (b,H,W,C) numpy.array
            # adv_images = adv_images.detach().cpu().numpy() * 255 # (b,H,W,C) numpy.array

            if save_img:
                # save image
                for j in range(len(adv_images)):
                    _ = save_adv_image(adv_images[j], img_name[j], save_root)
            
        att_acc = att_correct/num_all*100
        att_acc_total += att_acc
        nat_acc = nat_correct/num_all*100
        nat_acc_total += nat_acc

        print(" Acc for class {}: nat_acc: {}, att_acc: {}".format(cls, nat_acc, att_acc))
    
    print(" Total Acc: nat_acc: {}, att_acc: {}".format(nat_acc_total/num_class, att_acc_total/num_class))
        
    print('Attack compete')

def run_normal_cls_save_dict(atk_method, dataset_name, batch_size, base_data_root, adv_data_root, model, img_shape, params, save_img, device):
    # save: torch.pt, normalized adversarial input tensor

    transform_raw_to_clf = raw_to_clf()
    print('\n Attack dataset ', dataset_name)
    normalize = {}
    normalize['mean'] = (0.5, 0.5, 0.5)
    normalize['std'] = (0.5, 0.5, 0.5)

    transform = transforms.Compose([transforms.ToTensor()])
    trainroot = cifar10_root
    traindataset = tv.datasets.CIFAR10(root=trainroot, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=False)
    
    atk = get_atk_method(atk_method, model, params, normalize, trainloader)
    print(' Attack algorithm ', str(atk))
    
    if save_img:
        adv_input_dict = {}
        label_dict = {}
        img_name_dict = {}
        sub_root = atk_method + '_eps' + str(params['eps']) + '.pt'
        save_root = os.path.join(adv_data_root, sub_root)
        
        if os.path.exists(save_root):
            print("File {} already exists!".format(save_root))
            return 

    class_names = os.listdir(base_data_root)
    num_class = len(class_names)
    att_acc_total = 0
    nat_acc_total = 0
    for cls in class_names:
        if save_img:
            adv_input_list = []
            label_list = []
            img_name_list = []
        
        cls_root = os.path.join(base_data_root, cls)
        label = CIFAR10_LABEL[cls]
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = ImageList_Dataset(cls_root, transform, label)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_all = 0
        att_correct = 0
        nat_correct = 0
        for batch, (X_batch, y_batch, img_name) in enumerate(test_dataloader):

            num_all += X_batch.shape[0]

            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            adv_images = atk.__call__(X_batch, y_batch) # (b,C,H,W) normalized torch.tensor

            logits_nat = model(transform_raw_to_clf(X_batch))
            nat_correct_batch = torch.eq(torch.argmax(logits_nat, dim=1), y_batch).float().to('cpu').sum()
            
            logits_att = model(transform_raw_to_clf(adv_images))
            att_correct_batch = torch.eq(torch.argmax(logits_att, dim=1), y_batch).float().to('cpu').sum()

            att_correct += att_correct_batch
            nat_correct += nat_correct_batch

            if save_img:
                adv_input_list.append(adv_images.detach().cpu())
                label_list.append(y_batch.detach().cpu())
                img_name_list += img_name
                
        if save_img:  
            adv_input_list_cat = torch.cat(adv_input_list, 0)
            label_list_cat = torch.cat(label_list, 0)
            adv_input_dict[cls] = adv_input_list_cat
            label_dict[cls] = label_list_cat
            img_name_dict[cls] = img_name_list
            
        att_acc = att_correct/num_all*100
        att_acc_total += att_acc
        nat_acc = nat_correct/num_all*100
        nat_acc_total += nat_acc

        print(" Acc for class {}: nat_acc: {}, att_acc: {}".format(cls, nat_acc, att_acc))  
          
    if save_img:  
        save_dict = {
            "adv_inputs": adv_input_dict,
            "labels": label_dict,
            "img_names": img_name_dict,
            "nat_acc": nat_acc,
            "att_acc": att_acc,
        }  # nopep8
        torch.save(save_dict, save_root)
            
    print(" Total Acc: nat_acc: {}, att_acc: {}".format(nat_acc_total/num_class, att_acc_total/num_class))
        
    print('Attack compete')

def run_normal_cls_save_pic(atk_method, dataset_name, batch_size, base_data_root, adv_data_root, model, img_shape, params, save_img, device):
    # save: torch.pt, normalized adversarial input tensor

    transform_raw_to_clf = raw_to_clf()
    print('\n Attack dataset ', dataset_name)
    normalize = {}
    normalize['mean'] = (0.5, 0.5, 0.5)
    normalize['std'] = (0.5, 0.5, 0.5)

    transform = transforms.Compose([transforms.ToTensor()])
    trainroot = cifar10_root
    traindataset = tv.datasets.CIFAR10(root=trainroot, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=False)
    
    atk = get_atk_method(atk_method, model, params, normalize, trainloader)
    print(' Attack algorithm ', str(atk))

    class_names = os.listdir(base_data_root)
    num_class = len(class_names)
    att_acc_total = 0
    nat_acc_total = 0
    for cls in class_names:
        if save_img:
            adv_input_list = []
            label_list = []
        
        sub_root = 'advs/' + atk_method + '/eps' + str(params['eps'])
        save_root = os.path.join(adv_data_root, sub_root, cls)
        if not os.path.exists(save_root):
            os.makedirs(save_root) # 创建数据保存路径
        else:
            return
        
        cls_root = os.path.join(base_data_root, cls)
        label = CIFAR10_LABEL[cls]
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = ImageList_Dataset(cls_root, transform, label)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_all = 0
        att_correct = 0
        nat_correct = 0
        for batch, (X_batch, y_batch, img_name) in enumerate(test_dataloader):

            num_all += X_batch.shape[0]

            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            adv_images = atk.__call__(X_batch, y_batch) # (b,C,H,W) normalized torch.tensor

            logits_nat = model(transform_raw_to_clf(X_batch))
            nat_correct_batch = torch.eq(torch.argmax(logits_nat, dim=1), y_batch).float().to('cpu').sum()
            
            logits_att = model(transform_raw_to_clf(adv_images))
            att_correct_batch = torch.eq(torch.argmax(logits_att, dim=1), y_batch).float().to('cpu').sum()

            att_correct += att_correct_batch
            nat_correct += nat_correct_batch
            
            # adv_images = inverse_normalize(adv_images)
            adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() * 255 # (b,H,W,C) numpy.array
            # adv_images = adv_images.detach().cpu().numpy() * 255 # (b,H,W,C) numpy.array

            if save_img:
                # save image
                for j in range(len(adv_images)):
                    _ = save_adv_image(adv_images[j], img_name[j], save_root)

        att_acc = att_correct/num_all*100
        att_acc_total += att_acc
        nat_acc = nat_correct/num_all*100
        nat_acc_total += nat_acc

        print(" Acc for class {}: nat_acc: {}, att_acc: {}".format(cls, nat_acc, att_acc))  
            
    print(" Total Acc: nat_acc: {}, att_acc: {}".format(nat_acc_total/num_class, att_acc_total/num_class))
        
    print('Attack compete')
    
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

        
def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="training & validation batch size")
    parser.add_argument("--save_img", type=str2bool, default=True,
                        help="save images or not")
    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        help="base dataset name")
    parser.add_argument("--split", type=str, default='test',
                        help="training set or testing set")
    parser.add_argument("--attack_list", type=str, default=['FGSM'],
                        help="adv attack list ")
    parser.add_argument("--eps", type=str, default=8,
                        help="epsilon")
    parser.add_argument("--alpha", type=str, default=2,
                        help="alpha")
    parser.add_argument("--steps", type=str, default=10,
                        help="steps")
    parser.add_argument("--gamma", type=str, default=0,
                        help="gamma")

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def raw_to_clf():
    # CIFAR10
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose(
        [
            transforms.Normalize(mean, std)
        ]
    )
    return transform

def run_cifar10(args):
    dataset_name = args.dataset
    batch_size = args.batch_size
    save_img = args.save_img
    attack_list = args.attack_list
    device = torch.device('cuda')
    if args.split == 'train':
        base_data_root = normal_train_root
        adv_data_root = adv_train_pt_root # saving dir
    elif  args.split == 'test':
        base_data_root = normal_test_root 
        adv_data_root = adv_test_pt_root # saving dir

    img_shape = (32, 32) # 32 for cifar10
    
    eps = args.eps
    alpha = args.alpha
    steps = args.steps 
    gamma = args.gamma

    # get clf network and load saved weights
    model = WideResNet(num_classes=10).to(device)
    states_att = torch.load(defense_model_root, map_location="cuda:0") # Temporary t7 setting
    model.load_state_dict(states_att)
    model.eval()

    time0 = time.time()

    for atk_method in attack_list:
        params = {'eps':eps, 'alpha':alpha, 'steps':steps, 'gamma':gamma}
        time1 = time.time()
        run_normal_cls_save_dict(atk_method, dataset_name, batch_size, base_data_root, adv_data_root, model, img_shape, params, save_img, device)
        time2 = time.time()
        print('Time: ', time2 - time1)

    time2 = time.time()
    print('Total time: ', time2 - time0)
  
def main():
    args = input_args()
    dataset_name = args.dataset
    if dataset_name == 'CIFAR10':
        run_cifar10(args)
    else:
        raise Exception

if __name__=='__main__':
    main()
    # test()