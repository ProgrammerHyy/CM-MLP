import os
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data_path, skull=False):
        """
        :param data_path: path you store images and labels data
        :arg
            map_list: a index, containing the pathway of all images and labels and patient information
            train_list / val_list / test_list: contains filenames of data
        """
        self.image_path = os.path.join(data_path, "images")
        self.label_path = os.path.join(data_path, "labels")
        self.all_list = []

        for home, dirs, files in os.walk(self.image_path):
            for file in files:
                self.all_list.append(file)

        train_list, test_list = train_test_split(self.all_list, test_size=0.20)
        _, val_list = train_test_split(train_list, test_size=0.125)
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list

    def get_train_list(self):
        return self.train_list

    def get_val_list(self):
        return self.val_list

    def get_test_list(self):
        return self.test_list

    def get_all_list(self):
        return self.all_list

    def get_path(self):
        path = {
            'image_path': self.image_path,
            'label_path': self.label_path
                }
        return path
