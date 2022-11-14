from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras
from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange, reduce, repeat

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

    def call(self, x):
        '''
        n: number of ppl
        s: frame of each gait
        h: height
        w: width
        c: channel
        '''

        n, s, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
        x = rearrange(x, 'n s h w c -> (n s) h w c')
        # x = tf.reshape(x, (-1, h, w, c))
 
        x = self.foward_block(x)
        if self.pooling:
            x = self.pool2D(x)
        
        _, h, w, c =  tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # return x
        return rearrange(x, '(n s) h w c -> n s h w c', s= 16)


class HPM(layers.Layer):
    def __init__(self, in_dim, out_dim, bin_level_num=5): 
        super(HPM, self).__init__()
        self.bin_num = [2**i for i in range(bin_level_num)]
        self.fc_bin = tf.Variable(tf.keras.initializers.GlorotUniform()(
            (sum(self.bin_num), in_dim, out_dim)
        ), trainable=True)

    def call(self, x):
        feature = list()
        n, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        for num_bin in self.bin_num:
            z = tf.reshape(x, (n, num_bin, -1, c))
            z = tf.reduce_mean(z, axis = [2]) +  tf.reduce_max(z, axis = [2])
            feature.append(z)
        
        feature = tf.concat(feature, axis = 1)
        feature = tf.transpose(feature, perm=[1,0,2])
        feature = tf.matmul(feature, self.fc_bin) 
        
        return tf.transpose(feature, perm=[1, 0, 2])
        