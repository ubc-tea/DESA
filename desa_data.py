import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import random
import os

import fedbn_data_utils as data_utils

from cifar10c_dataset import CIFAR10C, CIFAR10C_preprocessed

import make_dirichlet_dataset


def compute_img_mean_std(img_set, im_size):
    means = torch.tensor([0.0, 0.0, 0.0])
    vars = torch.tensor([0.0, 0.0, 0.0])
    count = len(img_set) * im_size[0] * im_size[1]
    for i in range(len(img_set)):
        img, label = img_set[i]
        means += img.sum(axis        = [1, 2])
        vars += (img**2).sum(axis        = [1, 2])

    total_means = means / count
    total_vars  = (vars / count) - (total_means ** 2)
    total_stds  = torch.sqrt(total_vars)

    return total_means, total_stds


def prepare_data(args, im_size):

    if args.dataset == 'digits':

        MEANS = [[0.1307, 0.1307, 0.1307], [0.4379, 0.4440, 0.4731], [0.2473, 0.2473, 0.2473], [0.4828, 0.4603, 0.4320], [0.4595, 0.4629, 0.4097]]
        STDS = [[0.3015, 0.3015, 0.3015], [0.1161, 0.1192, 0.1017], [0.2665, 0.2665, 0.2665], [0.1960, 0.1938, 0.1977], [0.1727, 0.1603, 0.1785]]

        # Prepare data
        transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
            ])
        unnormalized_transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
            ])
        unnormalized_transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
            ])
        unnormalized_transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        transform_synth = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
            ])
        unnormalized_transform_synth = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        transform_mnistm = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[4], STDS[4])
            ])
        unnormalized_transform_mnistm = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])
        # MNIST
        unnormalized_mnist_trainset     = data_utils.DigitsDataset(data_path="../DeSAB/digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=True,  transform=unnormalized_transform_mnist)
        mnist_trainset     = data_utils.DigitsDataset(data_path="../DeSAB/digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
        mnist_testset      = data_utils.DigitsDataset(data_path="../DeSAB/digit_data/digit_dataset/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
        # unnormalized_mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=unnormalized_transform_mnist, download=True)
        # mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=transform_mnist, download=True)
        # mnist_testset = torchvision.datasets.MNIST(root="./digit_data", train=False, transform=transform_mnist, download=True)
        # print(f'MNIST: {len(mnist_testset)}')

        # SVHN
        unnormalized_svhn_trainset      = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=True,  transform=unnormalized_transform_svhn)
        svhn_trainset      = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
        svhn_testset       = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
        # unnormalized_svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=unnormalized_transform_svhn, download=True)
        # svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=transform_svhn, download=True)
        # svhn_testset = torchvision.datasets.SVHN(root="./digit_data", split='test', transform=transform_svhn, download=True)
        # print(f'SVHN: {len(svhn_testset)}')

        # USPS
        unnormalized_usps_trainset      = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=True,  transform=unnormalized_transform_usps)
        usps_trainset      = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
        usps_testset       = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
        # unnormalized_usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=unnormalized_transform_usps, download=True)
        # usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=transform_usps, download=True)
        # usps_testset = torchvision.datasets.USPS(root="./digit_data", train=False, transform=transform_usps, download=True)
        # print(f'USPS: {len(usps_testset)}')

        # Synth Digits
        unnormalized_synth_trainset     = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=unnormalized_transform_synth)
        synth_trainset     = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
        synth_testset      = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
        # unnormalized_synth_trainset     = ImageFolder('./digit_data/synthetic_digits/imgs_train', transform=unnormalized_transform_synth)
        # synth_trainset     = ImageFolder('./digit_data/synthetic_digits/imgs_train', transform=transform_synth)
        # synth_testset     = ImageFolder('./digit_data/synthetic_digits/imgs_valid', transform=transform_synth)
        # print(f'SYNTH: {len(synth_testset)}')

        # MNIST-M
        unnormalized_mnistm_trainset     = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=unnormalized_transform_mnistm)
        mnistm_trainset     = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
        mnistm_testset      = data_utils.DigitsDataset(data_path='../DeSAB/digit_data/digit_dataset/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
        # unnormalized_mnistm_trainset     = ImageFolder('./digit_data/mnistm/train', transform=unnormalized_transform_mnistm)
        # mnistm_trainset     = ImageFolder('./digit_data/mnistm/train', transform=transform_mnistm)
        # mnistm_testset     = ImageFolder('./digit_data/mnistm/test', transform=transform_mnistm)
        # print(f'MNISTM: {len(mnistm_testset)}')

        mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
        svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
        usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
        synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

        train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
        test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
        unnormalized_train_datasets = [unnormalized_mnist_trainset, unnormalized_svhn_trainset, unnormalized_usps_trainset, unnormalized_synth_trainset, unnormalized_mnistm_trainset]
        train_datasets = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
        test_datasets = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

        min_data_len = min(len(mnist_testset), len(svhn_testset), len(usps_testset), len(synth_testset), len(mnistm_testset))

    elif args.dataset == 'office':

        MEANS = [[0.7794, 0.7764, 0.7790], [0.6369, 0.6255, 0.6251], [0.4794, 0.4545, 0.3985], [0.6096, 0.6205, 0.6177]]
        STDS = [[0.3221, 0.3223, 0.3208], [0.3293, 0.3310, 0.3329], [0.2193, 0.2052, 0.2138], [0.2617, 0.2667, 0.2660]]

        data_base_path = './data'

        transform_amazon = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
        ])

        transform_caltech = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
        ])

        transform_dslr = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
        ])

        transform_webcam = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
        ])

        
        
        # amazon
        amazon_trainset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_amazon)
        amazon_testset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_amazon, train=False)
        # caltech
        caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_caltech)
        caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_caltech, train=False)
        # dslr
        dslr_trainset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_dslr)
        dslr_testset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_dslr, train=False)
        # webcam
        webcam_trainset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_webcam)
        webcam_testset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_webcam, train=False)

        # amazon_mean, amazon_std = compute_img_mean_std(amazon_trainset, im_size)
        # caltech_mean, caltech_std = compute_img_mean_std(caltech_trainset, im_size)
        # dslr_mean, dslr_std = compute_img_mean_std(dslr_trainset, im_size)
        # webcam_mean, webcam_std = compute_img_mean_std(webcam_trainset, im_size)

        # print(amazon_mean, amazon_std)
        # print(caltech_mean, caltech_std)
        # print(dslr_mean, dslr_std)
        # print(webcam_mean, webcam_std)
        
        # min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
        # val_len = int(min_data_len * 0.4)
        # min_data_len = int(min_data_len * 0.5)

        # amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
        # amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

        # caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
        # caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

        # dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
        # dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

        # webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
        # webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

        amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True)
        # amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=args.batch, shuffle=False)
        amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False)

        caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True)
        # caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=args.batch, shuffle=False)
        caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False)

        dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True)
        # dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=args.batch, shuffle=False)
        dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False)

        webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True)
        # webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=args.batch, shuffle=False)
        webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False)
        
        train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
        # val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
        test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
        # unnormalized_train_datasets = [unnormalized_amazon_trainset, unnormalized_svhn_trainset, unnormalized_usps_trainset, unnormalized_synth_trainset, unnormalized_mnistm_trainset]
        train_datasets = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
        test_datasets = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]

        min_data_len = min(len(amazon_testset), len(caltech_testset), len(dslr_testset), len(webcam_testset))

    elif args.dataset == 'retina':

        MEANS = [[0.5594, 0.2722, 0.0819], [0.7238, 0.3767, 0.1002], [0.5886, 0.2652, 0.1481], [0.7085, 0.4822, 0.3445]]
        STDS = [[0.1378, 0.0958, 0.0343], [0.1001, 0.1057, 0.0503], [0.1147, 0.0937, 0.0461], [0.1663, 0.1541, 0.1066]]

        # data_base_path = './data/segmented_retina'
        data_base_path = './data/retina_balanced'
        
        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor()
        ])
        
        # Drishti
        transform_drishti = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
        ])
        drishti_train_path = os.path.join(data_base_path, 'Drishti', 'Training')
        drishti_test_path = os.path.join(data_base_path, 'Drishti', 'Testing')
        unnormalized_drishti_trainset = ImageFolder(drishti_train_path, transform=transform_unnormalized)
        drishti_trainset = ImageFolder(drishti_train_path, transform=transform_drishti)
        drishti_testset = ImageFolder(drishti_test_path, transform=transform_drishti)
        
        # kaggle
        transform_kaggle = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
        ])
        kaggle_train_path = os.path.join(data_base_path, 'kaggle_arima', 'Training')
        kaggle_test_path = os.path.join(data_base_path, 'kaggle_arima', 'Testing')
        unnormalized_kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_unnormalized)
        kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_kaggle)
        kaggle_testset = ImageFolder(kaggle_test_path, transform=transform_kaggle)
        
        # RIM
        transform_rim = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
        ])
        rim_train_path = os.path.join(data_base_path, 'RIM', 'Training')
        rim_test_path = os.path.join(data_base_path, 'RIM', 'Testing')
        unnormalized_rim_trainset = ImageFolder(rim_train_path, transform=transform_unnormalized)
        rim_trainset = ImageFolder(rim_train_path, transform=transform_rim)
        rim_testset = ImageFolder(rim_test_path, transform=transform_rim)
        
        # refuge
        transform_refuge = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
        ])
        refuge_train_path = os.path.join(data_base_path, 'REFUGE', 'Training')
        refuge_test_path = os.path.join(data_base_path, 'REFUGE', 'Testing')
        unnormalized_refuge_trainset = ImageFolder(refuge_train_path, transform=transform_unnormalized)
        refuge_trainset = ImageFolder(refuge_train_path, transform=transform_refuge)
        refuge_testset = ImageFolder(refuge_test_path, transform=transform_refuge)
        

        #####################################
        Drishti_train_loader = torch.utils.data.DataLoader(drishti_trainset, batch_size=args.batch, shuffle=True)
        Drishti_test_loader = torch.utils.data.DataLoader(drishti_testset, batch_size=args.batch, shuffle=False)

        kaggle_train_loader = torch.utils.data.DataLoader(kaggle_trainset, batch_size=args.batch, shuffle=True)
        kaggle_test_loader = torch.utils.data.DataLoader(kaggle_testset, batch_size=args.batch, shuffle=False)

        rim_train_loader = torch.utils.data.DataLoader(rim_trainset, batch_size=args.batch, shuffle=True)
        rim_test_loader = torch.utils.data.DataLoader(rim_testset, batch_size=args.batch, shuffle=False)

        refuge_train_loader = torch.utils.data.DataLoader(refuge_trainset, batch_size=args.batch, shuffle=True)
        refuge_test_loader = torch.utils.data.DataLoader(refuge_testset, batch_size=args.batch, shuffle=False)
        
        train_loaders = [Drishti_train_loader, kaggle_train_loader, rim_train_loader, refuge_train_loader]
        test_loaders = [Drishti_test_loader, kaggle_test_loader, rim_test_loader, refuge_test_loader]
        unnormalized_train_datasets = [unnormalized_drishti_trainset, unnormalized_kaggle_trainset, unnormalized_rim_trainset, unnormalized_refuge_trainset]
        train_datasets = [drishti_trainset, kaggle_trainset, rim_trainset, refuge_trainset]
        test_datasets = [drishti_testset, kaggle_testset, rim_testset, refuge_testset]

        min_data_len = min(len(drishti_testset), len(kaggle_testset), len(rim_testset), len(refuge_testset))

    elif args.dataset == 'cifar10c':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    elif args.dataset == 'cifar10c_alpha1':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha1', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha1', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])
    
    elif args.dataset == 'cifar10c_alpha5':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha5', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha5', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    elif args.dataset == 'cifar10-0.2':
        MEANS = [[0, 0, 0] for _ in range(10)]
        STDS = [[1, 1, 1] for _ in range(10)]

        data_obj = make_dirichlet_dataset.DatasetObject(dataset='CIFAR10', n_client=10, unbalanced_sgm=0, rule='Dirichlet', rule_arg=0.2)

        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        
        for i in range(10):
            # print(data_obj.clnt_x[i].shape)
            # print(data_obj.tst_x[i].shape)
            trainset_tmp = make_dirichlet_dataset.Dataset(data_obj.clnt_x[i], data_obj.clnt_y[i].reshape(-1), train=True, dataset_name='CIFAR10')
            testset_tmp = make_dirichlet_dataset.Dataset(data_obj.tst_x[i], data_obj.tst_y[i].reshape(-1), train=False, dataset_name='CIFAR10')
            # trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = i, transform=transform_unnormalized)
            # testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])
    
    elif args.dataset == 'cifar10-0.5':
        MEANS = [[0, 0, 0] for _ in range(10)]
        STDS = [[1, 1, 1] for _ in range(10)]

        data_obj = make_dirichlet_dataset.DatasetObject(dataset='CIFAR10', n_client=10, unbalanced_sgm=0, rule='Dirichlet', rule_arg=0.5)

        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        
        for i in range(10):
            # print(data_obj.clnt_x[i].shape)
            # print(data_obj.tst_y[i].reshape(-1).shape)
            trainset_tmp = make_dirichlet_dataset.Dataset(data_obj.clnt_x[i], data_obj.clnt_y[i].reshape(-1), train=True, dataset_name='CIFAR10')
            testset_tmp = make_dirichlet_dataset.Dataset(data_obj.tst_x[i], data_obj.tst_y[i].reshape(-1), train=False, dataset_name='CIFAR10')
            # trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = i, transform=transform_unnormalized)
            # testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    elif args.dataset == 'cifar10-2':
        MEANS = [[0, 0, 0] for _ in range(10)]
        STDS = [[1, 1, 1] for _ in range(10)]

        data_obj = make_dirichlet_dataset.DatasetObject(dataset='CIFAR10', n_client=10, unbalanced_sgm=0, rule='Dirichlet', rule_arg=2)

        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        
        for i in range(10):
            # print(data_obj.clnt_x[i].shape)
            # print(data_obj.tst_y[i].reshape(-1).shape)
            trainset_tmp = make_dirichlet_dataset.Dataset(data_obj.clnt_x[i], data_obj.clnt_y[i].reshape(-1), train=True, dataset_name='CIFAR10')
            testset_tmp = make_dirichlet_dataset.Dataset(data_obj.tst_x[i], data_obj.tst_y[i].reshape(-1), train=False, dataset_name='CIFAR10')
            # trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = i, transform=transform_unnormalized)
            # testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    else:
        NotImplementedError


    shuffled_idxes = [list(range(0, len(test_datasets[idx]))) for idx in range(len(test_datasets))]
    for idx in range(len(shuffled_idxes)):
        random.shuffle(shuffled_idxes[idx])
    concated_test_set = [torch.utils.data.Subset(test_datasets[idx], shuffled_idxes[idx][:min_data_len]) for idx in range(len(test_datasets))]
    concated_test_set = torch.utils.data.ConcatDataset(concated_test_set)
    # print(len(concated_test_set))
    concated_test_loader = torch.utils.data.DataLoader(concated_test_set, batch_size=args.batch, shuffle=False)

    return train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, MEANS, STDS