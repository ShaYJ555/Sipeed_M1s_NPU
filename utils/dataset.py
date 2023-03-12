import os
import math
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from utils.transforms import Resize, RandomCrop, Normalize

class FaceKeyPointsDatasets(tf.keras.utils.Sequence):
    def __init__(self, batch_size, mode='train', grayscale=True):
        self.mode = mode
        if self.mode == "train":
            self.csv_file = "data/training_frames_keypoints.csv"
            self.data_path = "data/training"
        elif self.mode == 'test':
            self.csv_file = "data/test_frames_keypoints.csv"
            self.data_path = "data/test"
        self.df = pd.read_csv(self.csv_file, encoding='utf-8')
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.resize = Resize(256)
        self.crop = RandomCrop(240)
        self.gray = Normalize(scale=100)

    def __getitem__(self, index):
        image_list = []
        kpt_list = []
        for i in range(index*self.batch_size, (index+1)*self.batch_size):
            i = i % len(self.df)
            image_name = os.path.join(self.data_path, self.df.iloc[i, 0])
            # image = mpimg.imread(image_name)[:,:,0:3]
            image = Image.open(image_name).convert("RGB")
            image = np.array(image)
            kpt = self.df.iloc[i, 1:].values
            kpt = kpt.astype('float').reshape(-1)
            image, kpt = self.resize([image, kpt])
            image, kpt = self.crop([image, kpt])
            if self.grayscale:
                image, kpt = self.gray([image, kpt])
            image = np.array(image,dtype=np.uint8)
            kpt = np.array(kpt)
            image_list.append(image)
            kpt_list.append(kpt)
        return np.array(image_list), np.array(kpt_list)

    def __len__(self):
        return math.ceil(len(self.df) / float(self.batch_size))
