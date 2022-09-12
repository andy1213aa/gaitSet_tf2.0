from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras
from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras import layers


class BasicConv2D(layers.Layer):
    def __init__(self, filter_number, kernel_size, strides, **kwargs):
        super(BasicConv2D, self).__init__()

        self.conv2D = layers.Conv2D(
            filter_number, kernel_size, strides, use_bias=False, **kwargs)
        self.leaky_ReLU = layers.LeakyReLU()

    def call(self, input):
        x = self.conv2D(input)
        x = self.leaky_ReLU(x)
        return x


class SetBlock(layers.Layer):
    def __init__(self, foward_block, pooling=False):
        super(SetBlock, self).__init__()

        self.foward_block = foward_block
        self.pooling = pooling

        if pooling:
            self.pool2D = layers.MaxPooling2D()

    def call(self, input):
        '''
        n: number of ppl
        s: frame of each gait
        h: height
        w: width
        c: channel
        '''

        # n, s, h, w, c = input.shape
        # x = tf.reshape(input, (-1, h, w, c))
        # x = input.reshape((-1, h, w, c))
        x = self.foward_block(input)
        if self.pooling:
            x = self.pool2D(x)
        # _, h, w, c = x.shape
        return x
        # return tf.reshape(x, (-1, s, h, w, c))



