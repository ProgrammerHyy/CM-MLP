import os
import torch
import torch.utils.data as datas
import numpy as np
import albumentations as albu
import sys
sys.path.append('/home/myNet/')
from data.dataset import Dataset


class BrainDataset(datas.Dataset):

    def __init__(self, item_list, data_type, size, path, augmentations=True):
        '''
        :param item_list: a list contains items path in dataset
        :param data_type: type of your usage [train, test, val]
        :param augmentations: if you need data augmentation, default=True
        :param size: size of train images
        '''
        self.image_path = path['image_path']
        self.label_path = path['label_path']
        self.augmentation = augmentations
        self.size = size
        self.type = data_type
        self.id_list = item_list

        '''image&label data augmentation'''
        if self.augmentation:
            self.transformer_aug = albu.Compose(
                [
                    albu.ShiftScaleRotate(),
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                    albu.Resize(width=self.size, height=self.size)
                ]
            )
        else:
            self.transformer_aug = None
        self.transformer = albu.Compose([
            albu.Resize(width=self.size, height=self.size)
        ])

    def __getitem__(self, index):
        image = np.load(os.path.join(self.image_path, self.id_list[index])) / 255.0
        label = np.load(os.path.join(self.label_path, self.id_list[index]))
        if label.max() > 1:
            label[label > 1] = 1
        aug = None
        if self.type == 'train' and self.augmentation:
            aug = self.transformer_aug(image=image, mask=label)
        else:
            aug = self.transformer(image=image, mask=label)

        image, label = aug['image'], aug['mask']
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
            image = np.concatenate([image, image, image], axis=2)
        image = image.transpose(2, 0, 1).astype('float32')
        image = torch.tensor(image)

        label = label.transpose(2, 0, 1).astype('float32')
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return len(self.id_list)


def get_loader(data_root, batch_size, dtype, size, augmentations=True, shuffle=True, num_work=0, drop_last=True):

    dataset = Dataset(data_root)
    Data = None
    if dtype == 'train':
        Data = BrainDataset(dataset.get_train_list(), 'train', size, dataset.get_path(), augmentations)
    elif dtype == 'val':
        Data = BrainDataset(dataset.get_val_list(), 'val', size, dataset.get_path(), augmentations)
    elif dtype == 'test':
        Data = BrainDataset(dataset.get_test_list(), 'test', size, dataset.get_path(), augmentations)
    elif dtype == 'all':
        Data = BrainDataset(dataset.get_all_list(), 'all', size, dataset.get_path(), augmentations)
    data_loader = datas.DataLoader(
        dataset=Data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_work,
        drop_last=drop_last
    )
    return data_loader
