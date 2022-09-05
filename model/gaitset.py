
from typing import Set
import tensorflow as tf
from basic_block import BasicConv2D, SetBlock
from tensorflow.keras import layers


class GaitSet(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(GaitSet, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_frame = None

        # Set Network
        _set_channels = [32, 64, 128]

        self.set_layer1 = SetBlock(BasicConv2D(_set_channels[0], 5, (2,2), padding = 'same'))
        self.set_layer2 = SetBlock(BasicConv2D(_set_channels[0], 3, (2,2), padding = 'same'))
        self.set_layer3 = SetBlock(BasicConv2D(_set_channels[1], 3, (2,2), padding = 'same'))
        self.set_layer4 = SetBlock(BasicConv2D(_set_channels[1], 3, (2,2), padding = 'same'))
        self.set_layer5 = SetBlock(BasicConv2D(_set_channels[2], 3, (2,2), padding = 'same'))
        self.set_layer6 = SetBlock(BasicConv2D(_set_channels[2], 3, (2,2), padding = 'same'))

        # MGP
        _gl_channels = [64, 128]

        self.gl_layer1 = BasicConv2D(_gl_channels[0], 3, (2.2), padding = 'same')
        self.gl_layer2 = BasicConv2D(_gl_channels[0], 3, (2.2), padding = 'same')
        self.gl_layer3 = BasicConv2D(_gl_channels[1], 3, (2.2), padding = 'same')
        self.gl_layer4 = BasicConv2D(_gl_channels[1], 3, (2.2), padding = 'same')
        self.gl_pooling = layers.MaxPool2D()

        # HPM
        self.bin_num = [1, 2, 4, 8, 16]
        self.GAP = layers.GlobalAveragePooling2D()
        self.GMP = layers.GlobalMaxPool2D()

        # seperate FC in HPM
        self.Dense_layers_array = [self.layers.Dense(256, use_bias=False, name=f'FC_{idx}') for idx in range(sum(self.bin_num)*2)]
        
        



    def frame_max(self, x):

        if self.batch_frame is None:
            return tf.math.reduce_max(x)
        else:
            _tmp = [
                tf.math.reduce_max(x[:, self.batch_frame])
            ]


    def call(self, input):

        x = self.set_layer1(input)
        x = self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x))
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x))
        gl = self.gl_layer3(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)
        gl = gl + x

        feature_set_list = list()
        feature_MGP_list = list()
        n, h, w, c = gl.size()
        # HPP
        start = 0
        end = 0
        for bin in self.bin_num:
            
            end += bin
            # Set feature
            z = tf.reshape(x, (n, bin, -1, c))   
            feature_set = self.GAP(z) + self.GMP(z)
              # MGP feature
            z = tf.reshape(gl, (n, bin, h, -1, c))   
            feature_MGP = self.GAP(z) + self.GMP(z)

            for i in range(start, end):
                feature_set[:, i, :] = self.Dense_layers_array[i](feature_set[:, i, :])
                # The idx of MGP dense layer start from the half of the dense layer array's lens.
                feature_MGP[:, i, :] = self.Dense_layers_array[i + len(self.Dense_layers_array)//2](feature_set[:, i, :])

            feature_set_list.append(feature_set)
            feature_MGP_list.append(feature_MGP)

            start += bin

        feature_output = tf.concat([tf.convert_to_tensor(feature_MGP_list), tf.convert_to_tensor(feature_set_list)], axis=1)
        return feature_output
    

      
















