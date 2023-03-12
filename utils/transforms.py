import numpy as np
import tensorflow as tf

class Resize(object):
    # 将输入图像调整为指定大小

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        image = data[0]    # 获取图片
        key_pts = data[1]  # 获取标签
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        h, w = image_copy.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = tf.image.resize(image_copy, (new_h, new_w))

        # scale the pts, too
        key_pts_copy[::2] = key_pts_copy[::2] * new_w / w
        key_pts_copy[1::2] = key_pts_copy[1::2] * new_h / h

        return np.array(img), np.array(key_pts_copy)


class RandomCrop(object):
    # 随机位置裁剪输入的图像

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        h, w = image_copy.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_copy = image_copy[top: top + new_h,
                                left: left + new_w]

        key_pts_copy[::2] = key_pts_copy[::2] - left
        key_pts_copy[1::2] = key_pts_copy[1::2] - top

        return np.array(image_copy), np.array(key_pts_copy)

class Normalize(object):
    def __init__(self,scale) -> None:
        self.scale = scale

    def __call__(self, data):
        image = data[0]   # 获取图片
        key_pts = data[1]  # 获取标签
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        key_pts_copy = key_pts_copy / self.scale
        return np.array(image_copy,dtype=np.uint8),np.array(key_pts_copy)


class GrayNormalize(object):
    # # 将图片变为灰度图，并将其值放缩到[0, 1]
    # # 将 label 放缩到 [-1, 1] 之间
    def __init__(self,scale) -> None:
        self.scale = scale
    def __call__(self, data):
        image = data[0]   # 获取图片
        key_pts = data[1]  # 获取标签

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        # 灰度化图片
        image_copy = tf.image.rgb_to_grayscale(image_copy)

        key_pts_copy = key_pts_copy / self.scale
        
        return np.array(image_copy), np.array(key_pts_copy)
