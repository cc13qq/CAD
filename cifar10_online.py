# coding=utf-8

import torch
import numpy as np
import argparse
import os
import logging
import random

logger = logging.getLogger(__name__)


from config_cifar10 import *
import utils

from online_trainer_cad import Trainer
# ------------------------------------------------------------------------------
def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='The dataset')
    parser.add_argument('--attack_order', default=['PGD','SINIFGSM', 'BIM', 'RFGSM'], type=str)
    parser.add_argument('--mark', default='', type=str, help="")

    # train
    parser.add_argument('--clean_model', type=str, default='wrn-28-32')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='Input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='Input batch size for testing (default: 10)')
    parser.add_argument('--proto_batch_size', type=int, default=100, metavar='N',
                        help='Input batch size for training (default: 100)')
    parser.add_argument('--learning_rate', default=2.0e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=5.0e-6, type=float,
                        help='weight_decay')
    parser.add_argument('--lr_scheduler', default=False, type=utils.str2bool,
                        help='lr_scheduler')
    parser.add_argument('--epochs_base', default=200, type=int)
    parser.add_argument('--log_batch_base', default=100, type=int)
    parser.add_argument('--epochs_new', type=int, default=10)
    parser.add_argument('--lr_base', type=float, default=0.1)
    parser.add_argument('--lr_new', type=float, default=0.1)
    parser.add_argument('--base_class', default=10, type=int)
    parser.add_argument('--num_shot', default=10, type=int)
    parser.add_argument('--output_stages', type=int, default=None, help='output stages')

    # logging
    parser.add_argument('--log_dir', default='logs', type=str, help='path to save log')
    parser.add_argument('--exp_id', default= '00' , type=str, help='ID')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')


    ### CAD parameters
    # about pre-training
    parser.add_argument('-schedule', type=str, default='Cosine',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=200)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) 
    parser.add_argument('-new_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) 
    # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    parser.add_argument('-balance', type=float, default=0.001)
    parser.add_argument('-loss_iter', type=int, default=0)
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-eta', type=float, default=0.1)

    # about training
    parser.add_argument('--num_workers', type=int, default=8)


    args = parser.parse_args()
        
    if 'cifar10' in args.dataset:
        args.base_class = 10
    elif 'imagenet100' in args.dataset:
        args.base_class = 100
    args.num_way = args.base_class
    
    args.num_stage = len(args.attack_order)
    args.num_classes = args.base_class * args.num_stage
    
    if args.mark != '':
        args.mark = '_' + args.mark
    
    if args.output_stages is None:
        args.output_stages = args.num_stage

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args


if __name__ == '__main__':

    args = parse_args_and_config()

    exp_name = 'online_' + 'fewshot_' + args.exp_id + '_'  + args.mark
    args.log_dir = os.path.join(args.log_dir, exp_name)

    # set logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    logfile = os.path.join(args.log_dir, 'output_shot_'+str(args.num_shot)+'.log')
    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    
    logger.info(args)
    

    trainer = Trainer(args, logger)

    for stage in range(args.num_stage):
        if stage >= args.output_stages:
            break
        trainer.training(stage)

