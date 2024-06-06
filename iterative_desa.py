"""
Official implementation of DeSA: Overcoming Data and Model heterogeneities in Decentralized Federated Learning via Synthetic Anchors
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
import argparse
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from utils import get_network, get_time, TensorDataset
from condensation import distribution_matching, distribution_matching_DP
from torchvision.utils import save_image
import random
from loss_fn import Distance_loss
import pandas as pd
import torch.nn.functional as F
from PIL import Image

# for DP implementations
sys.path.append('./privacymaster/pyvacy/optim')
from pyvacymaster.pyvacy import optim as pyoptim
from pyvacymaster.pyvacy import analysis as pyanalysis
from pyvacymaster.pyvacy import sampling as pysampling

from desa_data import prepare_data



def GetPretrained(path, means, stds, im_size, num_classes, client_num, client_model_names, device, DP=False, ipc = 50, padding = 2):
    images_all = []
    for i in range(client_num):
        if DP:
            img_path = os.path.join(path, f"client{i}_{client_model_names[i]}_DM_{ipc}_DP_imgs.png")
        else:
            img_path = os.path.join(path, f'client{i}_{client_model_names[i]}_DM_{ipc}_imgs.png')
        images_pil = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Normalize(means[i], stds[i])
                ])
        images_torch = transform(images_pil)
        images = []
        for j in range(num_classes):
            for i in range(ipc):
                images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]].unsqueeze(0))
        images = torch.cat(images, dim=0).detach().to(device)
        # images.requires_grad = True
        images_all.append(images)
            
    return images_all

def calculate_kd_loss(y_pred_student, y_pred_teacher, y_true, loss_fn, temp=20., distil_weight=0.9):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.softmax(y_pred_teacher / temp, dim=1)
        soft_student_out = F.log_softmax(y_pred_student / temp, dim=1)

        loss = (1. - distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (distil_weight * temp * temp) * loss_fn(
            soft_student_out, soft_teacher_out
        )
        

        return loss

def train(model, train_loader, optimizer, loss_fun, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        _, output = model(x)

        loss = loss_fun(output, y)

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data


def train_vhl(model, optimizer, loss_fun, device, train_loader, virtual_loader, distance_loss, lambda_ori=1., lambda_reg=1.):
    model.train()
    correct = 0
    loss_all = 0
    align_loss_all = 0
    train_iter = iter(train_loader)
    virtual_iter = iter(virtual_loader)


    for step in range(len(train_iter)):

        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        client_features, output = model(x)

        classification_loss = loss_fun(output, y)

        align_loss = 0
        try:
            x_virtual, y_virtual = next(virtual_iter)
        except:
            virtual_iter = iter(virtual_loader)
            x_virtual, y_virtual = next(virtual_iter)
        virtual_feature = model.embed(x_virtual)
        align_loss = distance_loss(client_features, virtual_feature, y, y_virtual)

        
        # torch.autograd.set_detect_anomaly(True)
        loss = lambda_ori * classification_loss + lambda_reg * align_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        loss_all += loss.item()
        align_loss_all += align_loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/len(train_loader.dataset), align_loss_all/len(train_iter)


def train_kd(model, teacher_models, train_loader, virtual_loader, optimizer, kd_loss_fun, ce_loss_fun, device, lambda_ori=0.1, lambda_kd=1.):
    model.train()
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    virtual_iter = iter(virtual_loader)
    for step in range(len(train_iter)):

        # get ce loss
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        _, output = model(x)
        loss_ori = ce_loss_fun(output, y)
        
        # get kd loss
        try:
            x_virtual, y_virtual = next(virtual_iter)
        except:
            virtual_iter = iter(virtual_loader)
            x_virtual, y_virtual = next(virtual_iter)
        # num_data += y.size(0)
        x_virtual = x_virtual.to(device).float()
        y_virtual = y_virtual.to(device).long()
        # x = x.cuda(non_blocking=True).float()
        # y = y.cuda(non_blocking=True).long()
        _, virtual_output = model(x_virtual)
        output_targets = []
        with torch.no_grad():
            for teacher_model in teacher_models:
                _, output_target_tmp = teacher_model(x_virtual)
                output_targets.append(output_target_tmp)
            output_target = torch.mean(torch.stack(output_targets), dim=0)
            # output_target = output_target.cuda(non_blocking=True)
        # output_target = target_model(x)

        # loss = loss_fun(output, output_target)
        loss_kd = calculate_kd_loss(virtual_output, output_target, y_virtual, kd_loss_fun)

        loss = lambda_ori * loss_ori + lambda_kd * loss_kd
        

        optimizer.zero_grad()
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/len(train_loader.dataset)


def train_kd_vhl(client_list, model, example_logits, train_loader, kd_loader, reg_loader, optimizer, kd_loss_fun, ce_loss_fun, device, distance_loss, client_idx, it, lambda_ori=1., lambda_kd=1., lambda_reg=1.):
    model.train()
    correct = 0
    loss_all = 0
    loss_ori_all = 0
    loss_kd_all = 0
    loss_reg_all = 0
    train_iter = iter(train_loader)
    kd_iter = iter(kd_loader)
    kd_step = 0
    reg_iter = iter(reg_loader)
    for step in range(len(train_iter)):
        
        # get classification loss
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        features, output = model(x)
        loss_ori = ce_loss_fun(output, y)
        loss = loss_ori


        # get kd loss
        try:
            x_kd, y_kd = next(kd_iter)
        except:
            kd_iter = iter(kd_loader)
            x_kd, y_kd = next(kd_iter)
            kd_step = 0 # to make sure we get the correct logits from other clients
        x_kd = x_kd.to(device).float()
        y_kd = y_kd.to(device).long()

        _, kd_output = model(x_kd)
        # loss = loss_fun(output, output_target)
        teacher_logits = []
        for i, logits in enumerate(example_logits):
            if i in client_list and i != client_idx:
                teacher_logits.append(logits[kd_step])
        teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
        loss_kd = calculate_kd_loss(kd_output, teacher_logits, y_kd, kd_loss_fun)
        kd_step += 1


        # get regularization loss
        try:
            x_reg, y_reg = next(reg_iter)
        except:
            reg_iter = iter(reg_loader)
            x_reg, y_reg = next(reg_iter)
        x_reg = x_reg.to(device).float()
        y_reg = y_reg.to(device).long()
        reg_feature = model.embed(x_reg).detach()
        loss_reg = distance_loss(features, reg_feature, y, y_reg) # sup contrastive

        loss = lambda_ori * loss_ori + lambda_kd * loss_kd + lambda_reg * loss_reg
        loss_kd_all += loss_kd.item()
        

        optimizer.zero_grad()
        loss.backward()
        loss_all += loss.item()
        loss_ori_all += loss_ori.item()
        loss_reg_all += loss_reg.item()
        optimizer.step()
        
        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), loss_ori_all/len(train_iter), loss_kd_all/len(train_iter), loss_reg_all/len(train_iter), correct/len(train_loader.dataset)

def get_averaged_digits(teacher_models, virtual_loader, device, client_list):
    
    virtual_iter = iter(virtual_loader)
    output_targets = [[] for _ in teacher_models]
    for step in range(len(virtual_iter)):

        
        x_virtual, _ = next(virtual_iter)
        
        x_virtual = x_virtual.to(device).float()
        # y_virtual = y_virtual.to(device).long()

        
        with torch.no_grad():
            # for i, teacher_model in enumerate(teacher_models):
            for i in client_list:
                teacher_model = teacher_models[i]
                teacher_model.eval()
                _, output_target_tmp = teacher_model(x_virtual)
                output_targets[i].append(output_target_tmp)
            # output_target = torch.mean(torch.stack(output_targets), dim=0)


    return output_targets


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        _, output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)



def get_images(images_all, indices_class, c, n): # get random n images from class c
    if n < len(indices_class[c]):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
    else:
        idx_shuffle_0 = np.random.permutation(indices_class[c])
        idx_shuffle_1 = np.random.permutation(indices_class[c])[:n-len(indices_class[c])]
        idx_shuffle = np.concatenate([idx_shuffle_0, idx_shuffle_1], axis=0)
    return images_all[idx_shuffle]




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_net', type=float, default=1e-2, help='learning rate for models')
    parser.add_argument('--lr_kd', type=float, default=1e-2, help='learning rate for kd')
    parser.add_argument('--lr_img', type = float, default=5e-2, help = 'learning rate for img')
    parser.add_argument('--batch', type = int, default=32, help ='batch size')
    parser.add_argument('--kd_batch', type = int, default=None, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'target model training iterations')
    parser.add_argument('--c_iters', type = int, default=1, help = 'client training iterations')
    parser.add_argument('--inv_iters', type = int, default=1000, help = 'inversion training iterations')
    parser.add_argument('--kd_iters', type = int, default=100, help = 'knowledge distillation iterations')
    parser.add_argument('--save_path', type = str, default='./checkpoint', help='path to save the checkpoint')
    
    parser.add_argument('--ipc', type = int, default=50, help = 'sampled noisy images per class')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--dataset', type=str, default='digits', help='Dataset: digits')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--seed', type = int, default=0, help = 'random seeds')

    parser.add_argument('--init', type = str, default='normal', help='initialization method for noisy images')

    parser.add_argument('--kd', type = bool, default=False, help='knowledge distillation')
    parser.add_argument('--kd_from_scratch', type = bool, default=True, help='knowledge distillation from scratch or not')
    parser.add_argument('--second_wave', type = bool, default=False, help='2nd wave train model')
    parser.add_argument('--pretrain', type = bool, default=False, help='pretrain local model')
    parser.add_argument('--generate_image', type = bool, default=False, help='generate virtual image or not')
    parser.add_argument('--test', type = bool, default=False, help='test trained models')
    parser.add_argument('--resume', type = bool, default=False, help='resume from previous training')

    # parser.add_argument('--deep_inversion', type = bool, default=False, help='perform deep inversion to get global virtual data')
    # parser.add_argument('--DM', type = bool, default=False, help='perform distribution matching to get global virtual data')
    # parser.add_argument('--gen_method', type = str, default='DM', help='DM|inverted')
    parser.add_argument('--DP', type = bool, default=False, help='DP or not')

    parser.add_argument('--client_ratio', type = float, default=1.0, help = 'client sampling ratio')

    parser.add_argument('--lambda_ori', type=float, default=1., help='lambda for classification loss on original dataset')
    parser.add_argument('--lambda_kd', type=float, default=1., help='lambda for KD loss')
    parser.add_argument('--lambda_reg', type=float, default=1., help='lambda for regularization loss')

    parser.add_argument('--model_hetero', type = bool, default=True, help='whether the models are heterogeneous')

    args = parser.parse_args()
    print(args)
    args.device = device

    
    if args.kd_batch is None:
        args.kd_batch = args.batch
    
    if args.DP:
        assert (args.dataset == 'digits' and args.percent == 1.), 'Only support DP for digits with 100 percent data usage'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    torch.cuda.manual_seed_all(args.seed) 
    random.seed(args.seed)

    # prepare folder
    SAVE_PATH = os.path.join(args.save_path, args.dataset)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
   



    # prepare the data
    if args.dataset == 'digits':
        datasets = ['MNIST', 'SVHN', 'USPS', 'SynDigits', 'MNIST-M']
        if args.model_hetero:
            client_model_names = ['ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet']
            # client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet']
        else:
            client_model_names = ['ConvNet' for _ in datasets]
        num_classes, channel = 10, 3
        im_size = (32, 32)
    elif args.dataset == 'office':
        datasets = ['amazon', 'caltech', 'dslr', 'webcam']
        if args.model_hetero:
            # client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet']
            client_model_names = ['ConvNet', 'AlexNet', 'ConvNet', 'AlexNet']
        else:
            client_model_names = ['ConvNet' for _ in datasets]
        num_classes, channel = 10, 3
        im_size = (32, 32)
    elif args.dataset == 'cifar10c':
        datasets = [f'client{i}' for i in range(57)]
        if args.model_hetero:
            client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet']
        else:
            client_model_names = ['ConvNet' for _ in datasets]
        num_classes, channel = 10, 3
        im_size = (32, 32)
    elif args.dataset == 'cifar10-0.2':
        datasets = [f'client{i}' for i in range(10)]
        if args.model_hetero:
            client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet']
        else:
            client_model_names = ['ConvNet' for _ in datasets]
        num_classes, channel = 10, 3
        im_size = (32, 32)
    elif args.dataset == 'cifar10-0.5':
        datasets = [f'client{i}' for i in range(10)]
        if args.model_hetero:
            client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet']
        else:
            client_model_names = ['ConvNet' for _ in datasets]
        num_classes, channel = 10, 3
        im_size = (32, 32)
    elif args.dataset == 'cifar10-2':
        datasets = [f'client{i}' for i in range(10)]
        if args.model_hetero:
            client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 
                                'AlexNet', 'ConvNet', 'AlexNet', 'ConvNet']
        else:
            client_model_names = ['ConvNet' for _ in datasets]
        num_classes, channel = 10, 3
        im_size = (32, 32)
    else:
        raise NotImplementedError
    
    train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, MEANS, STDS = prepare_data(args, im_size)
    client_num = len(datasets)



    for i, dataset in enumerate(datasets):
        print(dataset)
        print(f'    Train: {len(train_datasets[i])}; Test: {len(test_datasets[i])}')

    # make save dictionary
    train_loss_save, reg_loss_save, kd_loss_save, train_acc_save, val_loss_save, val_acc_save, inter_loss_save, inter_acc_save = {}, {}, {}, {}, {}, {}, {}, {}
    if args.client_ratio == 1.:
        for client_idx in range(client_num):
            train_loss_save[f'Client{client_idx}'] = []
            reg_loss_save[f'Client{client_idx}'] = []
            kd_loss_save[f'Client{client_idx}'] = []
            train_acc_save[f'Client{client_idx}'] = []
            val_loss_save[f'Client{client_idx}'] = []
            val_acc_save[f'Client{client_idx}'] = []
            inter_loss_save[f'Client{client_idx}'] = []
            inter_acc_save[f'Client{client_idx}'] = []
    for client_idx in range(client_num):
        val_loss_save[f'Client{client_idx}'] = []
        val_acc_save[f'Client{client_idx}'] = []
        inter_loss_save[f'Client{client_idx}'] = []
        inter_acc_save[f'Client{client_idx}'] = []
        
    train_loss_save[f'mean'] = []
    reg_loss_save[f'mean'] = []
    kd_loss_save[f'mean'] = []
    train_acc_save[f'mean'] = []
    val_loss_save[f'mean'] = []
    val_acc_save[f'mean'] = []
    
    
    

    ''' Pretrain/Load local models '''
    client_models_pre = [get_network(client_model_name, channel, num_classes, im_size).to(args.device) for client_model_name in client_model_names]
    optimizers_pre = [optim.SGD(params=client_models_pre[i].parameters(), lr=args.lr_net) for i in range(len(client_models_pre))]
    classification_loss_fun = nn.CrossEntropyLoss()
    if args.pretrain:
        print('Pretrain local models')
        for client_idx in range(client_num):
            for i in range(0, args.iters):
                loss, acc = train(client_models_pre[client_idx], train_loaders[client_idx], optimizers_pre[client_idx], classification_loss_fun, device)
                test_loss, test_acc = test(client_models_pre[client_idx], test_loaders[client_idx], classification_loss_fun, device)

                if (i+1) % 10 == 0:
                    print('Client {}'.format(client_idx))
                    print('Train|  Epoch {} - Loss: {:4f}; Acc: {:4f}'.format(i, loss, acc))
                    print('Test |  Epoch {} - Loss: {:4f}; Acc: {:4f}'.format(i, test_loss, test_acc))

            ''' Save checkpoint '''
            print(' Saving checkpoints to {}...'.format(SAVE_PATH))
            torch.save(client_models_pre[client_idx].state_dict(), f'{SAVE_PATH}/client{client_idx}_pretrained_{client_model_names[client_idx]}_model.pt')
    else:
        print('Load local models')
        for client_idx in range(client_num):
            client_models_pre[client_idx].load_state_dict(torch.load(f'{SAVE_PATH}/client{client_idx}_pretrained_{client_model_names[client_idx]}_model.pt'))
    
    # To avoid changing BN statistics
    for i in range(len(client_models_pre)):
        client_models_pre[i].eval()
    
    '''Train/Load virtual data'''
    label_syns_tmp = torch.tensor(np.array([np.ones(args.ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    image_syns = [torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device) for idx in range(client_num)]
    label_syns = [copy.deepcopy(label_syns_tmp).to(args.device) for idx in range(client_num)]
    
    

    data_path = f'{SAVE_PATH}'

    # deep inversion
    if args.generate_image:
        print('Start distribution matching...')

        for client_idx in range(client_num):
            # organize the real dataset
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(num_classes)]
            images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
            labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)
            images_all = torch.cat(images_all, dim=0).to(args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
            
            # print(image_batch)

            optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=1, momentum=0.5) # optimizer_img for synthetic data
            inv_iters = args.inv_iters
            image_batch = 256

            if args.DP:
                min_image_batch = min([len(indices_class_) for indices_class_ in indices_class])
                # image_batch = min(image_batch, 1024)
                image_batch = min(256, min_image_batch)
                dpsgd_params = {
                    'l2_norm_clip' : 2.,
                    'noise_multiplier' : 0.6,
                    'minibatch_size' : image_batch,
                    'microbatch_size' : args.ipc,
                    'lr' : 1,
                    'momentum' : 0.5
                }
                print(dpsgd_params)
                # optimizer_img_dp = pyoptim.DPSGD(params=[image_syns[client_idx], ], 
                #                               **dpsgd_params) # optimizer_img for synthetic data
                optimizer_img_dp = pyoptim.DPSGD(params=[image_syns[client_idx], ], **dpsgd_params) # optimizer_img for synthetic data
                
                inv_iters = int(args.inv_iters/(min_image_batch*num_classes/image_batch))
                print(inv_iters)
                minibatch_loader, microbatch_loader = pysampling.get_data_loaders(
                    dpsgd_params['minibatch_size'],
                    dpsgd_params['microbatch_size'],
                    inv_iters
                )
                
                # DELTA = 1/len(train_datasets[client_idx])
                DELTA = 1/(min_image_batch*num_classes)
                print('Achieves ({}, {})-DP'.format(
                    pyanalysis.epsilon(
                        min_image_batch*num_classes,
                        dpsgd_params['minibatch_size'],
                        dpsgd_params['noise_multiplier'],
                        inv_iters,
                        DELTA
                    ),
                    DELTA
                ))
                # sys.exit()
            
            for it in range(inv_iters):
                loss_avg = 0
                if args.DP:
                    # get real images for each class
                    image_real = [get_images(images_all, indices_class, c, min_image_batch) for c in range(num_classes)]
                    loss, image_syns[client_idx] = distribution_matching_DP(image_real, image_syns[client_idx], optimizer_img_dp, channel, num_classes, im_size, args.ipc, minibatch_loader, microbatch_loader)
                else:
                    # get real images for each class
                    image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(num_classes)]
                    # print([image_real[i].size(0) for i in range(len(image_real))])
                    loss, image_syns[client_idx] = distribution_matching(image_real, image_syns[client_idx], optimizer_img, channel, num_classes, im_size, args.ipc)
                # report averaged loss
                loss_avg += loss
                loss_avg /= num_classes
                if it%100 == 0:
                    print('%s Initialization:\t client = %2d, iter = %05d, loss = %.4f' % (get_time(), client_idx, it, loss_avg))
       

        ''' Save generated data '''
        print(' Saving generated data to {}'.format(SAVE_PATH))
        for i, local_syn_images in enumerate(image_syns):
            if args.DP:
                save_name = os.path.join(data_path, f'client{i}_{client_model_names[i]}_DM_{args.ipc}_DP_imgs.png')
            else:
                save_name = os.path.join(data_path, f'client{i}_{client_model_names[i]}_DM_{args.ipc}_imgs.png')
            
            image_syn_vis = copy.deepcopy(local_syn_images.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * STDS[i][ch] + MEANS[i][ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            save_image(image_syn_vis, save_name, nrow=args.ipc)
    else:
        print('Load virtual data...')
        image_syns = GetPretrained(data_path, MEANS, STDS, im_size, num_classes, client_num, client_model_names, args.device, DP = args.DP, ipc = args.ipc)


    # ''' Test inverted data  '''
    # virtual_test_loss = dict()
    # virtual_test_acc = dict()
    # for i in range(len(image_syns)):
    #     virtual_test_loss[i] = []
    #     virtual_test_acc[i] = []
    #     image_syn_eval_id = copy.deepcopy(image_syns[i].detach().to(device))
    #     label_syn_eval_id = copy.deepcopy(label_syns[i].detach().to(device))
    #     # virtual_train_set = TensorDataset(image_syn_eval, label_syn_eval)
    #     # virtual_train_loader = torch.utils.data.DataLoader(virtual_train_set, batch_size=args.batch, shuffle=True, num_workers=0)
        
    #     for j in range(len(image_syns)):
    #         image_syn_eval_ood = copy.deepcopy(image_syns[j].detach().to(device))
    #         label_syn_eval_ood = copy.deepcopy(label_syns[j].detach().to(device))
    #         image_syn_eval = (image_syn_eval_id + image_syn_eval_ood)/2
    #         label_syn_eval = label_syn_eval_id
    #         virtual_train_set = TensorDataset(image_syn_eval, label_syn_eval)
    #         virtual_train_loader = torch.utils.data.DataLoader(virtual_train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    #         val_loss, val_acc = test(client_models_pre[j], virtual_train_loader, classification_loss_fun, device)
    #         virtual_test_loss[i].append(val_loss)
    #         virtual_test_acc[i].append(val_acc)


    ''' Prepare mixup vitual data '''
    # get global virtual data
    global_virtual_images = [copy.deepcopy(image_syns[client_idx].detach().cpu()).to(args.device) for client_idx in range(client_num)]
    global_virtual_labels = [copy.deepcopy(label_syns[client_idx].detach().cpu()).to(args.device) for client_idx in range(client_num)]
    # global_virtual_image = torch.cat(global_virtual_images, dim=0)
    # global_virtual_label = torch.cat(global_virtual_labels, dim=0)
    # global_virtual_image_cuda = global_virtual_image.to(args.device)
    # global_virtual_label_cuda = global_virtual_label.to(args.device)
    # global_train_set = TensorDataset(global_virtual_image, global_virtual_label)
    # global_train_loader = torch.utils.data.DataLoader(global_train_set, batch_size=args.kd_batch, shuffle=True, num_workers=0)  

    # # calculate data weighting for each client
    # data_mixup_ratio = [[] for _ in range(client_num)]
    # for client_idx in range(client_num):
    #     # organize the real dataset
    #     images_all = []
    #     labels_all = []
    #     indices_class = [[] for c in range(num_classes)]
    #     images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
    #     labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
    #     for i, lab in enumerate(labels_all):
    #         indices_class[lab].append(i)
    #     for i in range(len(indices_class)):
    #         data_mixup_ratio[client_idx].append(len(indices_class[i]))
    # data_mixup_ratio = np.array(data_mixup_ratio)
    # data_mixup_ratio = data_mixup_ratio/data_mixup_ratio.sum(axis=0)

    # mixup images
    mixup_virtual_images = torch.mean(torch.stack(global_virtual_images), dim=0).detach().cpu()
    mixup_virtual_labels = global_virtual_labels[0].detach().cpu()
    mixup_train_set = TensorDataset(mixup_virtual_images, mixup_virtual_labels)
    shuffled_idx = list(range(0, len(mixup_train_set)))
    random.shuffle(shuffled_idx)
    shuffled_mixup_train_set = torch.utils.data.Subset(mixup_train_set, shuffled_idx[:len(mixup_train_set)])
    kd_train_loader = torch.utils.data.DataLoader(shuffled_mixup_train_set, batch_size=args.kd_batch, shuffle=False, num_workers=0)
    reg_train_loader = torch.utils.data.DataLoader(shuffled_mixup_train_set, batch_size=args.kd_batch, shuffle=True, num_workers=0)

    # concatenated train sets
    concated_train_sets = [torch.utils.data.ConcatDataset([train_dataset, mixup_train_set]) for train_dataset in train_datasets]
    concated_train_loaders = [torch.utils.data.DataLoader(concated_train_set, batch_size=args.kd_batch, shuffle=True, num_workers=0) for concated_train_set in concated_train_sets]

    ''' Knowledge Distillation '''
    distance_loss = Distance_loss(device=args.device)
    
    # prepare model and optimizer
    if args.kd_from_scratch:
        client_models_kd = [get_network(client_model_name, channel, num_classes, im_size).to(args.device) for client_model_name in client_model_names]
    else:
        client_models_kd = [copy.deepcopy(client_model_pre) for client_model_pre in client_models_pre]
    # optimizers_kd = [optim.Adam(params=client_models_kd[i].parameters(), lr=args.lr_kd) for i in range(len(client_models_kd))]
    optimizers_kd = [optim.SGD(params=client_models_kd[i].parameters(), lr=args.lr_kd) for i in range(len(client_models_kd))]
    kd_loss_fun = nn.KLDivLoss()

    if args.kd:
        print('Start KD...')
        if args.resume:
            print('Load kd models')
            for client_idx in range(client_num):
                if args.DP:
                    if args.client_ratio != 1:
                        model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_DP_{client_model_names[client_idx]}_{args.client_ratio}_model.pt'
                    else:
                        model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_DP_{client_model_names[client_idx]}_model.pt'
                else:
                    if args.client_ratio != 1:
                        model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_{client_model_names[client_idx]}_{args.client_ratio}_model.pt'
                    else:
                        model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_{client_model_names[client_idx]}_model.pt'
                client_models_kd[client_idx].load_state_dict(torch.load(model_path))
        for i in range(args.kd_iters):
            # get clients
            if args.client_ratio != 1.:
                client_list = np.random.choice(np.arange(client_num), int(args.client_ratio*client_num), replace=False)
            else:
                client_list = np.arange(client_num)
            print(F'Selected {int(args.client_ratio*client_num)} clients for round {i}:')
            print(client_list)

            # get averaged logits
            output_logits = get_averaged_digits(client_models_kd, kd_train_loader, device, client_list)

            if (i+1) % 10 == 0:
                print('----------')
            tr_mean_loss, reg_mean_loss, kd_mean_loss, tr_mean_acc, te_mean_loss, te_mean_acc = [], [], [], [], [], []
            for client_idx in client_list:
                for c_iter in range(args.c_iters):
                    # print(averaged_logits)
                    # loss, acc = train_kd(client_models_kd[client_idx], client_models, train_loaders[client_idx], mixup_train_loader, optimizers_kd[client_idx], kd_loss_fun, classification_loss_fun, args.device, args.lambda_ori, args.lambda_kd)
                    loss, loss_ori, loss_kd, loss_reg, acc = train_kd_vhl(client_list, client_models_kd[client_idx], output_logits, concated_train_loaders[client_idx], kd_train_loader, reg_train_loader, optimizers_kd[client_idx], kd_loss_fun, classification_loss_fun, device, distance_loss, client_idx, i, lambda_ori=args.lambda_ori, lambda_kd=args.lambda_kd, lambda_reg=args.lambda_reg)
                    test_loss, test_acc = test(client_models_kd[client_idx], test_loaders[client_idx], classification_loss_fun, args.device)
                if (i+1) % 10 == 0:
                    print('Epoch {}: KD Train|  Client {} - Loss: {:4f}; Ori Loss: {:4f}; KD Loss: {:4f}; Reg Loss: {:4f}; Acc: {:4f}'.format(i, client_idx, loss, loss_ori, loss_kd, loss_reg, acc))
                    print('Epoch {}: KD Test |  Client {} - Loss: {:4f}; Acc: {:4f}'.format(i, client_idx, test_loss, test_acc))
                '''
                # save trianing outcome
                if args.client_ratio == 1.:
                    train_loss_save[f'Client{client_idx}'].append(loss_ori)
                    reg_loss_save[f'Client{client_idx}'].append(loss_reg)
                    kd_loss_save[f'Client{client_idx}'].append(loss_kd)
                    train_acc_save[f'Client{client_idx}'].append(acc)
                    
                tr_mean_loss.append(loss_ori)
                reg_mean_loss.append(loss_reg)
                kd_mean_loss.append(loss_kd)
                tr_mean_acc.append(acc)
                
                if client_idx == client_list[-1]:
                    train_loss_save['mean'].append(np.mean(tr_mean_loss))
                    reg_loss_save['mean'].append(np.mean(reg_mean_loss))
                    kd_loss_save['mean'].append(np.mean(kd_mean_loss))
                    train_acc_save['mean'].append(np.mean(tr_mean_acc))
                '''
                    

            ''' Record inter loss and acc after each global iteration '''
            '''
            te_mean_loss, te_mean_acc = [], []
            for client_idx in range(client_num):
                kd_test_loss, kd_test_acc = test(client_models_kd[client_idx], test_loaders[client_idx], classification_loss_fun, args.device)
                
                val_loss_save[f'Client{client_idx}'].append(test_loss)
                val_acc_save[f'Client{client_idx}'].append(test_acc)
                
                te_mean_loss.append(test_loss)
                te_mean_acc.append(test_acc)
                
                if client_idx == client_num-1:
                    val_loss_save['mean'].append(np.mean(te_mean_loss))
                    val_acc_save['mean'].append(np.mean(te_mean_acc))
                
                avg_kd_loss, avg_kd_acc = [], []
                for client_j in range(client_num):
                    if client_idx != client_j:
                        kd_test_loss_j, kd_test_acc_j = test(client_models_kd[client_idx], test_loaders[client_j], classification_loss_fun, args.device)
                        avg_kd_loss.append(kd_test_loss_j)
                        avg_kd_acc.append(kd_test_acc_j)

                avg_kd_loss = np.mean(avg_kd_loss)
                avg_kd_acc = np.mean(avg_kd_acc)
                inter_loss_save[f'Client{client_idx}'].append(np.mean(avg_kd_loss))
                inter_acc_save[f'Client{client_idx}'].append(np.mean(avg_kd_acc))
            '''



        ''' Save checkpoint '''
        for client_idx in range(client_num):
            if args.DP:
                if args.client_ratio != 1:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_DP_{client_model_names[client_idx]}_{args.client_ratio}_model.pt'
                else:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_DP_{client_model_names[client_idx]}_model.pt'
            else:
                if args.client_ratio != 1:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_{client_model_names[client_idx]}_{args.client_ratio}_model.pt'
                else:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_{client_model_names[client_idx]}_model.pt'
            print(' Saving checkpoints to {}...'.format(model_path))
            torch.save(client_models_kd[client_idx].state_dict(), model_path)
    else:
        print('Load kd models')
        for client_idx in range(client_num):
            if args.DP:
                if args.client_ratio != 1:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_DP_{client_model_names[client_idx]}_{args.client_ratio}_model.pt'
                else:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_DP_{client_model_names[client_idx]}_model.pt'
            else:
                if args.client_ratio != 1:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_{client_model_names[client_idx]}_{args.client_ratio}_model.pt'
                else:
                    model_path = f'{SAVE_PATH}/client{client_idx}_iterative_kd_{client_model_names[client_idx]}_model.pt'
            client_models_kd[client_idx].load_state_dict(torch.load(model_path))



    ''' Final testing '''
    pre_intra_mean, pre_inter_mean, kd_intra_mean, kd_inter_mean = [], [], [], []
    pre_worst_acc = 1
    kd_worst_acc = 1

    # pre_global_acc = []
    # pre_global_loss = []
    # kd_global_acc = []
    # kd_global_loss = []
    # for client_idx in range(client_num):
    #     pre_test_loss, pre_test_acc = test(client_models_pre[client_idx], concated_test_loader, classification_loss_fun, args.device)
    #     kd_test_loss, kd_test_acc = test(client_models_kd[client_idx], concated_test_loader, classification_loss_fun, args.device)

    #     pre_global_acc.append(pre_test_acc)
    #     pre_global_loss.append(pre_test_loss)
    #     kd_global_acc.append(kd_test_acc)
    #     kd_global_loss.append(kd_test_loss)
        
    #     print(f'Client {client_idx}: {datasets[client_idx]} with {client_model_names[client_idx]}')
    #     print('PRE Test |  Loss: {:4f}; Acc: {:4f};'.format(pre_test_loss, pre_test_acc))
    #     print('KD  Test |  Loss: {:4f}; Acc: {:4f};'.format(kd_test_loss, kd_test_acc))
    #     if client_idx == client_num-1:
    #         print('Pre  Test | Global Acc: {:4f}; Global Loss: {:4f};'.format(np.mean(pre_global_acc), np.mean(pre_global_loss)))
    #         print('KD  Test | GLobal Acc: {:4f}; Global Loss: {:4f};'.format(np.mean(kd_global_acc), np.mean(kd_global_loss)))

    for client_idx in range(client_num):
        pre_test_loss, pre_test_acc = test(client_models_pre[client_idx], test_loaders[client_idx], classification_loss_fun, args.device)
        kd_test_loss, kd_test_acc = test(client_models_kd[client_idx], test_loaders[client_idx], classification_loss_fun, args.device)
        avg_pre_acc, avg_kd_acc = [], []
        for client_j in range(client_num):
            if client_idx != client_j:
                _, pre_test_acc_j = test(client_models_pre[client_idx], test_loaders[client_j], classification_loss_fun, args.device)
                _, kd_test_acc_j = test(client_models_kd[client_idx], test_loaders[client_j], classification_loss_fun, args.device)
                avg_pre_acc.append(pre_test_acc_j)
                avg_kd_acc.append(kd_test_acc_j)
        
        if pre_worst_acc > pre_test_acc:
            pre_worst_acc = pre_test_acc
        if kd_worst_acc > kd_test_acc:
            kd_worst_acc = kd_test_acc
        
        avg_pre_acc = np.mean(avg_pre_acc)
        avg_kd_acc = np.mean(avg_kd_acc)
        
        pre_intra_mean.append(pre_test_acc)
        pre_inter_mean.append(avg_pre_acc)
        kd_intra_mean.append(kd_test_acc)
        kd_inter_mean.append(avg_kd_acc)
        print(f'Client {client_idx}: {datasets[client_idx]} with {client_model_names[client_idx]}')
        print('PRE Test |  Loss: {:4f}; Acc: {:4f}; Avg. OOD Acc:{:4f}'.format(pre_test_loss, pre_test_acc, avg_pre_acc))
        print('KD  Test |  Loss: {:4f}; Acc: {:4f}; Avg. OOD Acc:{:4f}'.format(kd_test_loss, kd_test_acc, avg_kd_acc))
        if client_idx == client_num-1:
            print('Pre  Test | Intra Acc: {:4f}; Inter Acc: {:4f}; worst: {:4f}'.format(np.mean(pre_intra_mean), np.mean(pre_inter_mean), pre_worst_acc))
            print('KD  Test | Intra Acc: {:4f}; Inter Acc: {:4f}; worst: {:4f}'.format(np.mean(kd_intra_mean), np.mean(kd_inter_mean), kd_worst_acc))
    
    # print('Virtual test loss:')
    # print(virtual_test_loss)
    # print('Virtual test acc:')
    # print(virtual_test_acc)

    # # # Save acc and loss results
    # model_type = 'heterogeneous' if args.model_hetero else 'homogeneous'
    # DP = 'DP' if args.DP else 'clean'
    # metrics_pd = pd.DataFrame.from_dict(train_loss_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_train_loss_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(reg_loss_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_reg_loss_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(kd_loss_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_kd_loss_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(train_acc_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_train_acc_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(val_loss_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_val_loss_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(val_acc_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_val_acc_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(inter_loss_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_inter_loss_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))
    # metrics_pd = pd.DataFrame.from_dict(inter_acc_save)
    # metrics_pd.to_csv(os.path.join(SAVE_PATH,f"desab_{model_type}_{args.dataset}_inter_acc_IPC{args.ipc}_{DP}_{args.lambda_reg}_{args.lambda_kd}_{args.client_ratio}_{args.seed}.csv"))

