import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import time
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
from copy import deepcopy
import networks
from config_cifar10 import *
import utils
import dataset
from Network import MYNET

# tiny
class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.start_stage = 0

        self.dt, self.ft = utils.Averager(), utils.Averager()
        self.bt, self.ot = utils.Averager(), utils.Averager()
        self.timer = utils.Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.num_stage
        
        self.attack_order = args.attack_order
        _, test_datasets = dataset.get_dataset_all_pt(args, self.attack_order, w_clean = False)
        self.test_dataloaders_adv = {}
        for attack in test_datasets:
            self.test_dataloaders_adv[attack] = DataLoader(test_datasets[attack], batch_size=self.args.test_batch_size, shuffle=False)
        
        _, test_datasets = dataset.get_dataset_all_pt(args, self.attack_order, w_clean = True)
        self.test_dataloaders_all = {}
        for attack in test_datasets:
            self.test_dataloaders_all[attack] = DataLoader(test_datasets[attack], batch_size=self.args.test_batch_size, shuffle=False)


        self.device = torch.device("cuda")

        self.num_classes = args.num_classes
        self.num_shot = args.num_shot

        # Define model and optimizer
        self.clean_model, self.trans_clean = utils.get_model(args, 'wrn-28-32', self.device)
        self.clean_model.eval()

        self.model = MYNET(self.args, mode=self.args.base_mode).to(self.device)
        self.model = nn.DataParallel(self.model)
        self.model.train()

        self.old_model = None

        self.prototype = None
        self.class_label = None

        self.dummy_classifiers = None
        self.old_classifiers = None
        
    
    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler


    def training(self, stage):
        self.attack_new = self.attack_order[stage]
        self.logger.info("\nStage: {} Attack: {}".format(stage, self.attack_new))
        
        if stage == 0:
            self.training_base(stage)
        else:
            self.training_few_shot(stage)
        
        if stage == 0 :
            train_dataset_base, _ = dataset.get_dataset_session0_pt(self.args, self.attack_new)
            train_dataloader = DataLoader(train_dataset_base, batch_size=self.args.batch_size, shuffle=True)
            proto_train_dataset_base, _ = dataset.get_dataset_session0_pt(self.args, self.attack_new, proto=True)
            proto_dataloader = DataLoader(proto_train_dataset_base, batch_size=self.args.proto_batch_size, shuffle=True)
        else:
            train_dataset_fewshot, _ = dataset.get_dataset_fewshot_pt(self.args, self.attack_new, expand=stage, num_shot=self.args.num_shot)
            train_dataloader= DataLoader(train_dataset_fewshot, batch_size=self.args.batch_size, shuffle=True)
            proto_train_dataset_fewshot, _ = dataset.get_dataset_fewshot_pt(self.args, self.attack_new, expand=stage, num_shot=self.args.num_shot, proto=True)
            proto_dataloader= DataLoader(proto_train_dataset_fewshot, batch_size=self.args.proto_batch_size, shuffle=True)

        # load best model from last stage
        if stage == 0:
            save_dir_pre = os.path.join(self.args.log_dir, 'model_defense_'+str(stage)+'.pth')
        else:
            save_dir_pre = os.path.join(self.args.log_dir, 'model_defense_'+str(stage)+'_shot_'+str(self.args.num_shot)+'.pth')
        pretrained_model = torch.load(save_dir_pre)['state_dict']
        self.model.load_state_dict(pretrained_model, strict=False)
        self.logger.info("Loading defense model from {}".format(save_dir_pre))

        ### save prototype
        self.protoSave(self.model, train_dataloader, stage)
        
        if self.dummy_classifiers is None :
            #save dummy classifiers
            self.dummy_classifiers=deepcopy(self.model.module.fc.weight.detach())
            
            self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
            self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]
        
        if stage >= 0:
            self.test_defense_adv1(stage, self.test_dataloaders_adv)
    
    def training_base(self, stage):
        
        self.save_dir = os.path.join(self.args.log_dir, 'model_defense_'+str(stage)+'.pth')
        if os.path.exists(self.save_dir):
            self.logger.info("Defense Model exist: {}".format(self.save_dir))
            pretrained_model = torch.load(self.save_dir)['state_dict']
            self.model.load_state_dict(pretrained_model, strict=False)
            self.logger.info("Loading defense model from {}".format(self.save_dir))
            return

        train_dataset_base, test_dataset_base = dataset.get_dataset_session0_pt(self.args, self.attack_new)
        train_dataloader_base= DataLoader(train_dataset_base, batch_size=self.args.batch_size, shuffle=True)
        test_dataloader_base = DataLoader(test_dataset_base, batch_size=self.args.test_batch_size, shuffle=False)
        self.logger.info("Base Train num: {} Test num: {}".format(train_dataset_base.__len__(), test_dataset_base.__len__()))

        #gen_mask
        masknum = 3
        mask = np.zeros((self.args.base_class, self.args.num_classes))
        for i in range(self.args.num_classes - self.args.base_class):
            picked_dummy = np.random.choice(self.args.base_class, masknum, replace=False)
            mask[:, i + self.args.base_class][picked_dummy]=1
        mask = torch.tensor(mask).to(self.device)

        optimizer, scheduler = self.get_optimizer_base()

        for epoch in range(self.args.epochs_base):
            start_time = time.time()
            # train base sess
            tl, ta = self.base_train(self.model, train_dataloader_base, test_dataloader_base, optimizer, epoch, mask, self.save_dir)

            # test model with all seen class
            tsl, tsa = self.test(self.model, test_dataloader_base, epoch, stage)

            # save better model
            if (tsa * 100) >= self.trlog['max_acc'][stage]:
                self.trlog['max_acc'][stage] = float('%.3f' % (tsa * 100))
                self.trlog['max_acc_epoch'] = epoch

                torch.save({'state_dict':self.model.state_dict(),
                            'test_acc':tsa,}, self.save_dir)
                self.logger.info("Best Epoch {:d}: Best Test Acc {:.4f}".format(epoch, tsa * 100))
                self.logger.info('Saving model to :%s' % self.save_dir)

            self.trlog['train_loss'].append(tl)
            self.trlog['train_acc'].append(ta)
            self.trlog['test_loss'].append(tsl)
            self.trlog['test_acc'].append(tsa)
            scheduler.step()

        if not self.args.not_data_init:
            pretrained_model = torch.load(self.save_dir)['state_dict']
            self.model.load_state_dict(pretrained_model, strict=False)
            self.model = self.replace_base_fc(train_dataset_base, test_dataloader_base.dataset.transform, self.model)
            self.logger.info('Replace the fc with average embedding, and save it to :%s' % self.save_dir)
            torch.save({'state_dict':self.model.state_dict()}, self.save_dir)

            self.model.module.mode = 'avg_cos'
            tsl, tsa = self.test(self.model, test_dataloader_base, 0, stage)
            if (tsa * 100) >= self.trlog['max_acc'][stage]:
                self.trlog['max_acc'][stage] = float('%.3f' % (tsa * 100))
                self.logger.info('The new best test acc of base stage {:.3f}'.format(self.trlog['max_acc'][stage]))
            torch.save({'state_dict':self.model.state_dict(),
                            'test_acc':tsa,}, self.save_dir)

        #save dummy classifiers
        self.dummy_classifiers=deepcopy(self.model.module.fc.weight.detach())
        
        self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
        self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]

    def training_few_shot(self, stage):
        if stage == 0:
            self.save_dir = os.path.join(self.args.log_dir, 'model_defense_'+str(stage)+'.pth')
        else:
            self.save_dir = os.path.join(self.args.log_dir, 'model_defense_'+str(stage)+'_shot_'+str(self.args.num_shot)+'.pth')
        if os.path.exists(self.save_dir):
            self.logger.info("Defense Model exist: {}".format(self.save_dir))
            pretrained_model = torch.load(self.save_dir)['state_dict']
            self.model.load_state_dict(pretrained_model, strict=False)
            self.logger.info("Loading defense model from {}".format(self.save_dir))
            return
        
        train_dataset_fewshot, test_dataset_fewshot = dataset.get_dataset_fewshot_pt(self.args, self.attack_new, expand=stage, num_shot=self.args.num_shot)
        train_dataloader_fewshot= DataLoader(train_dataset_fewshot, batch_size=self.args.num_shot, shuffle=True)
        test_dataloader_fewshot = DataLoader(test_dataset_fewshot, batch_size=self.args.test_batch_size, shuffle=False)
        self.logger.info("Fewshot Train num: {} Test num: {}".format(train_dataset_fewshot.__len__(), test_dataset_fewshot.__len__()))

        self.model.module.mode = self.args.new_mode
        self.model.eval()
        
        class_list = [i + stage * 10 for i in range(self.args.base_class)]

        self.old_model = deepcopy(self.model)
        tl, ta = self.fewshot_train1(stage, train_dataloader_fewshot, test_dataloader_fewshot, class_list, self.save_dir)
                
        tsl, tsa = self.test(self.model, test_dataloader_fewshot, -1, stage)
        self.logger.info('Test Acc {:.3f}'.format(tsa*100))

    
    def test_defense_adv1(self, stage, test_dataloaders):
        self.logger.info("Testing defense model on all attacks")
        self.model.eval()

        attack = self.args.attack_order[stage]

        test_dataiter = iter(test_dataloaders[attack])
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        test_l = 0
        acc = 0
        test_n = 0
        test_loss = 0
        test_acc = 0

        for test_step in range(0, len(test_dataiter)):

            X, y, _ = next(test_dataiter)
            X = X.to(self.device)
            y = y.to(self.device)

            logits = self.model(X, stage=stage)
            logits_ = logits[:, :self.args.base_class*(stage + 1)]

            loss = criterion(logits_, y)

            test_l += loss.item() * y.size(0)
            acc += (logits_.max(1)[1] % 10 == y%10).sum().item()
            test_n += y.size(0)

            test_loss = test_l/test_n
            test_acc = acc/test_n

        test_time = time.time()

        self.logger.info("Stage {:d}: Attack {}, Test Time {:.1f}, Test Loss {:.4f}, Test Acc {:.4f}".format(stage, attack, test_time - start_time, test_loss, test_acc))

    
    def proto_augment(self, old_class):

        indexs = np.random.randint(0, old_class, size=(self.args.batch_size))

        proto_aug = self.prototype[indexs]
        proto_aug_label = self.class_label[indexs]
        proto_aug = torch.from_numpy(proto_aug).float().to(self.device)
        proto_aug_label = torch.from_numpy(proto_aug_label).to(self.device)

        return proto_aug, proto_aug_label
    
    def _compute_loss(self, imgs, target, old_class=0):
        if self.old_model is None:
            output = self.model(imgs)
            loss_cls = nn.CrossEntropyLoss()(output, target)
            return loss_cls
        else:
            output, feature = self.model(imgs, return_feature = True)
            _, feature_old = self.old_model(imgs, return_feature = True)

            loss_cls = nn.CrossEntropyLoss()(output, target)

            loss_kd = nn.CosineEmbeddingLoss()(feature, feature_old.detach(), torch.ones(imgs.shape[0]).to(self.device))

            proto_aug, proto_aug_label = self.proto_augment(old_class)
            soft_feat_aug = self.model.module.linear(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / 0.1, proto_aug_label)

            return loss_cls + loss_protoAug + loss_kd

    def protoSave(self, model, loader, stage):
        self.logger.info("Saving prototypes Stage {}".format(stage))
        if stage == 0:
            save_dir_proto = os.path.join(self.args.log_dir, 'prototype_'+str(stage)+'.npy')
            save_dir_clslabel = os.path.join(self.args.log_dir, 'class_label_'+str(stage)+'.npy')
        else:
            if self.args.num_attacks_per_stage > 1:
                save_dir_proto = os.path.join(self.args.log_dir, 'prototype_'+str(stage)+'_shot_'+str(self.args.num_shot)+'_attack_'+str(self.args.num_attacks_per_stage)+'.npy')
                save_dir_clslabel = os.path.join(self.args.log_dir, 'class_label_'+str(stage)+'_shot_'+str(self.args.num_shot)+'_attack_'+str(self.args.num_attacks_per_stage)+'.npy')
            else:
                save_dir_proto = os.path.join(self.args.log_dir, 'prototype_'+str(stage)+'_shot_'+str(self.args.num_shot)+'.npy')
                save_dir_clslabel = os.path.join(self.args.log_dir, 'class_label_'+str(stage)+'_shot_'+str(self.args.num_shot)+'.npy')

        if os.path.exists(save_dir_proto) and os.path.exists(save_dir_clslabel):
            self.logger.info("Prototypes exist: {}".format(save_dir_proto))
            self.prototype = np.load(save_dir_proto)
            self.class_label = np.load(save_dir_clslabel)
            self.logger.info("Loading prototypes from {}".format(save_dir_proto))
        else:
            features = []
            labels = []
            model.eval()
            dataiter = iter(loader)
            data_bar = tqdm(range(1, len(dataiter) + 1))
            with torch.no_grad():
                for i in data_bar:
                    X, y, _ = next(dataiter)
                    X = X.to(self.device)
                    feature = model.module.encode(X)
                    # if feature.shape[0] == self.args.batch_size:
                    labels.append(y.numpy())
                    features.append(feature.cpu().numpy())
            
            labels = np.concatenate(labels)
            labels_set = np.unique(labels)
            features = np.concatenate(features)
            
            prototype = []
            class_label = []
            for item in labels_set:
                index = np.where(item == labels)[0]
                class_label.append(item)
                feature_classwise = features[index]
                prototype.append(np.mean(feature_classwise, axis=0))

            if stage == 0:
                self.prototype = np.array(prototype)
                self.class_label = np.array(class_label)
            else:
                self.prototype = np.concatenate((prototype, self.prototype), axis=0)
                self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            
            # saving
            np.save(save_dir_proto, self.prototype)
            np.save(save_dir_clslabel, self.class_label)

    def base_train(self, model, trainloader, testloader, optimizer, epoch, mask, save_dir):
        stage = 0
        tl = utils.Averager()
        ta = utils.Averager()
        model = model.train()
        train_dataiter = iter(trainloader)
        train_bar = tqdm(range(1, len(train_dataiter) + 1))

        for train_step in train_bar:

            beta = torch.distributions.beta.Beta(self.args.alpha, self.args.alpha).sample([]).item()
            X, y, _ = next(train_dataiter)
            data = X.to(self.device)
            train_label = y.to(self.device)

            logits = model(data)
            logits_ = logits[:, :self.args.base_class]
            
            # print(logits.shape, logits_.shape)
            loss = F.cross_entropy(logits_, train_label)
            
            acc = utils.count_acc(logits_, train_label)
            
            
            if epoch >= self.args.loss_iter:
                logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
                logits_masked_chosen= logits_masked * mask[train_label]
                pseudo_label = torch.argmax(logits_masked_chosen[:, self.args.base_class:], dim=-1) + self.args.base_class
                #pseudo_label = torch.argmax(logits_masked[:,args.base_class:], dim=-1) + args.base_class
                loss2 = F.cross_entropy(logits_masked, pseudo_label)

                index = torch.randperm(data.size(0)).cuda()
                pre_emb1 = model.module.pre_encode(data)
                mixed_data = beta * pre_emb1 + (1 - beta) * pre_emb1[index]
                mixed_logits = model.module.post_encode(mixed_data)

                newys=train_label[index]
                idx_chosen=newys!=train_label
                mixed_logits=mixed_logits[idx_chosen]

                pseudo_label1 = torch.argmax(mixed_logits[:, self.args.base_class:], dim=-1) + self.args.base_class # new class label
                pseudo_label2 = torch.argmax(mixed_logits[:, :self.args.base_class], dim=-1)  # old class label
                loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
                novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
                loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
                total_loss = loss + self.args.balance*(loss2+loss3+loss4)

            else:
                total_loss = loss

            desc = 'Stage 0 Epoch {} Train Loss {:.4f} Train Acc {:.4f}'.format(epoch, total_loss.item(), acc)
            train_bar.set_description(desc)
            train_bar.update()
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if self.args.log_batch_base > 0 and train_step % self.args.log_batch_base == 0:
                _, tsa = self.test(model, testloader, epoch, stage)
                self.logger.info('Stage 0 Epoch {} Train Loss {:.4f} Train Acc {:.4f} Test Acc {:.4f}'.format(epoch, total_loss.item(), acc, tsa))
                model.train()

                if (tsa * 100) >= self.trlog['max_acc'][stage]:
                    self.trlog['max_acc'][stage] = float('%.3f' % (tsa * 100))
                    self.trlog['max_acc_epoch'] = epoch

                    torch.save({'state_dict':model.state_dict(),
                                'test_acc':tsa,}, save_dir)
                    self.logger.info("Best Epoch {:d}: Best Test Acc {:.4f}".format(epoch, tsa * 100))


        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def replace_base_fc(self, trainset, transform, model):
        # replace fc.weight with the embedding average of train data
        model = model.eval()

        trainloader = DataLoader(dataset=trainset, batch_size=self.args.batch_size, num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            train_dataiter = iter(trainloader)
            for test_step in range(0, len(train_dataiter)):

                data, label, _ = next(train_dataiter)
                embedding = model.module.encode(data.to(self.device))

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []

        for class_index in range(self.args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)

        model.module.fc.weight.data[:self.args.base_class] = proto_list

        return model

    def test(self, model, testloader, epoch, stage):
        test_class = self.args.base_class*(stage + 1)
        model = model.eval()
        vl = utils.Averager()
        va = utils.Averager()

        test_dataiter = iter(testloader)
        test_bar = tqdm(range(1, len(test_dataiter) + 1))

        with torch.no_grad():
            for test_step in test_bar:
                X, y, _ = next(test_dataiter)
                data = X.to(self.device)
                test_label = y.to(self.device)

                logits = model(data, stage=stage)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
                acc = utils.count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

            desc = 'Epoch {:03d} Test Loss {:.4f} Test Acc {:.4f}'.format(epoch, vl, va)
            test_bar.set_description(desc)
            test_bar.update()

        return vl, va

    def update_fc_avg(self, data, label, class_list):
        new_fc=[]
        for class_index in class_list:
            data_index = np.where(label == class_index)[0]
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.model.module.fc.weight.data[class_index]=proto
        new_fc = torch.stack(new_fc,dim=0)
        return new_fc
    
    def fewshot_train1(self, stage, trainloader, testloader, class_list, save_dir):
        old_class = class_list[0] - 1
        tl = utils.Averager()
        ta = utils.Averager()
        self.model.eval()

        data = []
        label = []
        for batch in trainloader:
            data_i, label_i, _ = batch
            data_i = self.model.module.encode(data_i.to(self.device)).detach()
            data.append(data_i)
            label.append(label_i)
        
        data = torch.cat(data, 0)
        label = torch.cat(label, 0)
        
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.model.module.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        self.model.eval()
        self.old_model.eval()
        
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        for epoch in range(self.args.epochs_new):

            for i, batch in enumerate(trainloader):
                X, y, _ = batch
                X = X.to(self.device)
                y = y.to(self.device)

                feature = self.model.module.encode(X)

                old_fc = self.model.module.fc.weight[:self.args.base_class * stage, :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.model.module.get_logits1(feature, fc, stage)
                logits_ = logits[:, :self.args.base_class*(stage + 1)]
                loss_cls = F.cross_entropy(logits_, y)

                proto_aug, proto_aug_label = self.proto_augment(old_class)
                soft_feat_aug = self.model.module.get_logits1(proto_aug, old_fc, stage)
                loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / 0.1, proto_aug_label)

                loss = loss_cls # + loss_protoAug 
                # print(loss_cls, loss_protoAug)
                # loss = loss_protoAug + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = utils.count_acc(logits_, y, mode='train')
                    
                tl.add(loss.item())
                ta.add(acc)

                self.model.module.fc.weight.data[self.args.base_class * stage: self.args.base_class * (stage + 1), :].copy_(new_fc.data)

            _, tsa = self.test(self.model, testloader, epoch, stage)
            # self.test_defense_adv(stage, self.test_dataloaders_adv)
            self.model.train()

            self.logger.info('Stage {} Epoch {} Train Loss {:.4f} Train Acc {:.4f} Test Acc {:.4f}'.format(stage, epoch, loss.item(), acc, tsa*100))

            if (tsa * 100) >= self.trlog['max_acc'][stage]:
                self.trlog['max_acc'][stage] = float('%.3f' % (tsa * 100))
                self.trlog['max_acc_epoch'] = epoch

                torch.save({'state_dict':self.model.state_dict(),
                            'test_acc':tsa,}, save_dir)
                self.logger.info("Best Epoch {:d}: Best Test Acc {:.4f}".format(epoch, tsa * 100))


        tl = tl.item()
        ta = ta.item()
        return tl, ta
    
