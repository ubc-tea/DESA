import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms


class CIFAR10C(Dataset):
    '''
    In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
    and the last 10,000 images are the test set images corrupted at severity five. labels.npy is the label file for all other image files.
    '''
    def __init__(self, site, base_path='./data', train=True, client_num=1, transform=None):
        assert site in ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                        'defocus_blur','glass_blur','motion_blur','zoom_blur',
                        'snow','frost','fog',
                        'brightness','contrast','elastic_transform','pixelate','jpeg_compression',
                        'speckle_noise','gaussian_blur','spatter','saturate'
                        ]
        self.base_path = base_path
        labels = np.load(f'{base_path}/CIFAR-10-C/labels.npy')

        data = np.load(f'{base_path}/CIFAR-10-C/{site}.npy')
        total_data = []
        total_label = []
        for level in range(1, 6):
            level_data = data[(level-1)*10000:level*10000]
            level_label = labels[(level-1)*10000:level*10000]
            ratio = 0.9
            splitpoint = int(level_data.shape[0]*ratio)
            if train:
                # total_data.append(np.asarray(level_data[(client_num-1)*int(splitpoint/3):client_num*int(splitpoint/3)]))
                # total_label.append(np.asarray(level_label[(client_num-1)*int(splitpoint/3):client_num*int(splitpoint/3)], dtype=np.long))
                total_data.append(np.asarray(level_data[:splitpoint]))
                total_label.append(np.asarray(level_label[:splitpoint], dtype=np.long))
                # print('image', np.asarray(level_data[:splitpoint]).shape)
                # print('label', np.asarray(level_label[:splitpoint], dtype=np.long).shape)
            else:
                total_data.append(np.asarray(level_data[splitpoint:]))
                total_label.append(np.asarray(level_label[splitpoint:], dtype=np.long))

        self.imgs = np.concatenate(total_data, axis=0)
        self.labels = np.concatenate(total_label, axis=0)
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        image = Image.fromarray(image).convert('RGB')

        # if len(image.split()) != 3:
        #     image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    


class CIFAR10C_preprocessed(Dataset):
    
    def __init__(self, base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = 0, transform=None):
        
        self.transform = transform

        if train:
            self.labels = np.asarray(np.load(f'{base_path}/train/client{client_num}_labels.npy'), dtype=np.long)
            self.images = np.asarray(np.load(f'{base_path}/train/client{client_num}_images.npy'))
        else:
            self.labels = np.asarray(np.load(f'{base_path}/test/client{client_num}_labels.npy'), dtype=np.long)
            self.images = np.asarray(np.load(f'{base_path}/test/client{client_num}_images.npy'))
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image).convert('RGB')
        label = torch.tensor(label, dtype=torch.long)

        # if len(image.split()) != 3:
        #     image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
            # transforms.Resize(im_size),            
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            transforms.ToTensor()
            ])
            image = transform(image)

        return image, label
    
# total_trainset = []
# for i in range(2):
#     trainset = CIFAR10C_preprocessed(client_num = i)
#     total_trainset.append(trainset)

# print(len(total_trainset[0]), len(total_trainset[1]))
# print(total_trainset[1][100][0].size())
# print(total_trainset[1][100][1])
# print(total_trainset[0].labels)
