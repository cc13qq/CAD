import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar10']:
            self.encoder = networks.WideResNet_backbone(num_classes=self.args.base_class)
            self.num_features = 640
        elif self.args.dataset in ['imagenet100']:
            self.encoder = networks.ResNet50_backbone(num_classes=self.args.base_class)
            self.num_features = 512*4
        # if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000']:
        #     self.encoder = resnet18(False, args)  # pretrained=False
        #     self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        
        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier=nn.Linear(self.num_features, self.pre_allocate-self.args.base_class, bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        
        self.dummy_orthogonal_classifier.weight.data=self.fc.weight.data[self.args.base_class:,:]
        
        # print(self.dummy_orthogonal_classifier.weight.data.size())
        # print('self.dummy_orthogonal_classifier.weight initialized over.')

    def forward_metric(self, x, return_feature=False, stage=0):
        x = self.encode(x)
        if 'cos' in self.mode:
            
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
            
            out = torch.cat([x1[:, :self.args.base_class * (stage+1)], x2],dim=1)
            # out = torch.cat([x1[:, self.args.base_class * stage: self.args.base_class * (stage+1)],x2],dim=1)
            
            out = self.args.temperature * out
            
        elif 'dot' in self.mode:
            out = self.fc(x)
            out = self.args.temperature * out
        
        if return_feature:
            return out, x
        else:
            return out

    def forpass_fc(self,x):
        x = self.encode(x)
        if 'cos' in self.mode:
            
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
    
    def pre_encode(self,x):
        
        if self.args.dataset in ['cifar10','cifar100','manyshotcifar']:
            x = self.encoder.conv1(x)
            # x = self.encoder.bn1(x)
            # x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            
        elif self.args.dataset in ['imagenet100','mini_imagenet','manyshotmini','cub200']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
        
        return x
        
    
    def post_encode(self,x):
        if self.args.dataset in ['cifar10','cifar100','manyshotcifar']:
            
            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['imagenet100','mini_imagenet','manyshotmini','cub200']:
            
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
            
        return x

    def forward(self, input, return_feature=False, stage=0):
        if self.mode != 'encoder':
            input = self.forward_metric(input, return_feature, stage)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')
    
    def get_fea_logit(self, x, return_feature=False):
        x = self.encode(x)
        logit = self.fc(x)
        if return_feature:
            return logit, x
        else:
            return logit

    def update_fc(self,dataloader,class_list,session, device):
        data = []
        label = []
        for batch in dataloader:
            data_i, label_i, _ = batch
            data_i = self.encode(data_i.to(device)).detach()
            data.append(data_i)
            label.append(label_i)
        
        data = torch.cat(data, 0)
        label = torch.cat(label, 0)

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session, device)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
    
    def get_logits1(self,x,fc, stage):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
            
            out = torch.cat([x1[:, :self.args.base_class * (stage+1)], x2],dim=1)
            out = self.args.temperature * out
            return out
    
    def get_logit_from_fea(self, x, stage=0):
        if 'cos' in self.mode:
            
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
            
            out = torch.cat([x1[:,:self.args.base_class * (stage+1)],x2],dim=1)
            
            out = self.args.temperature * out
            
        elif 'dot' in self.mode:
            out = self.fc(x)
            out = self.args.temperature * out
        
        return out

    def update_fc_ft(self,new_fc,data,label,session, device):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.num_way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.num_way * (session - 1):self.args.base_class + self.args.num_way * session, :].copy_(new_fc.data)

