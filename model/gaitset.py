
from typing import Set
import tensorflow as tf
from .basic_block import BasicConv2D, SetBlock
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


class GaitSet(tf.keras.Model):
    def __init__(self, hidden_dim=32):
        super(GaitSet, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_frame = None

        # Set Network
        _set_channels = [32, 64, 128]

        self.set_layer1 = SetBlock(BasicConv2D(
            _set_channels[0], 5, (1, 1), padding='same'))
        self.set_layer2 = SetBlock(BasicConv2D(
            _set_channels[0], 3, (2, 2), padding='same'))
        self.set_layer3 = SetBlock(BasicConv2D(
            _set_channels[1], 3, (1, 1), padding='same'))
        self.set_layer4 = SetBlock(BasicConv2D(
            _set_channels[1], 3, (2, 2), padding='same'))
        self.set_layer5 = SetBlock(BasicConv2D(
            _set_channels[2], 3, (1, 1), padding='same'))
        self.set_layer6 = SetBlock(BasicConv2D(
            _set_channels[2], 3, (1, 1), padding='same'))

        # MGP
        _gl_channels = [64, 128]

        self.gl_layer1 = BasicConv2D(
            _gl_channels[0], 3, (1, 1), padding='same')
        self.gl_layer2 = BasicConv2D(
            _gl_channels[0], 3, (1, 1), padding='same')
        self.gl_layer3 = BasicConv2D(
            _gl_channels[1], 3, (1, 1), padding='same')
        self.gl_layer4 = BasicConv2D(
            _gl_channels[1], 3, (1, 1), padding='same')
        self.gl_pooling = layers.MaxPool2D()

        # HPM
        self.bin_num = [1, 2, 4, 8, 16]
        self.GAP = layers.GlobalAveragePooling2D()
        self.GMP = layers.GlobalMaxPool2D()

        # seperate FC in HPM
        self.Dense_layers_array = [layers.Dense(
            256, use_bias=False, name=f'FC_{idx}') for idx in range(sum(self.bin_num)*2)]

    # def frame_max(self, x):

    #     n, h, w, c = x.shape
    #     k = 8 # number of ppl
    #     x = tf.reshape(x, (k, -1, h, w, c))
    #     x =
        # return tf.reduce_max(x, axis=)

    def call(self, input):

        x = self.set_layer1(input)
        x = self.set_layer2(x)
        # gl = self.gl_layer1(self.frame_max(x))
        gl = self.gl_layer1(x)
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        # gl = self.gl_layer3(gl + self.frame_max(x))
        gl = self.gl_layer3(gl)
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        # x = self.frame_max(x)

        gl = gl + x

        feature_Set_list = list()
        feature_MGP_list = list()

        n, h, w, c = gl.shape
        # HPP
        start = 0
        end = 0

        for bin in self.bin_num:

            end += bin
            # Set feature
            z = tf.reshape(x, (-1, bin, h//bin, w, c))
            feature_Set = tf.reduce_max(
                z, axis=[2, 3]) + tf.reduce_mean(z, axis=[2, 3])

            # MGP feature
            z = tf.reshape(gl, (-1, bin, h//bin, w, c))
            feature_MGP = tf.reduce_max(
                z, axis=[2, 3]) + tf.reduce_mean(z, axis=[2, 3])

            # print('feature_Set: ', feature_Set.shape)
            # print('feature_MGP: ', feature_MGP.shape)

            tmp_feature_Set = []
            tmp_feature_MGP = []

            for i in range(start, end):
                tmp_feature_Set.append(
                    self.Dense_layers_array[i](feature_Set[:, i-start]))
                '''
                The idx of MGP dense layer start from the half of the dense layer array's lens.
                '''
                tmp_feature_MGP.append(
                    self.Dense_layers_array[i + len(self.Dense_layers_array)//2](feature_MGP[:, i-start]))

            tmp_feature_Set = tf.stack(tmp_feature_Set, axis=1)
            tmp_feature_MGP = tf.stack(tmp_feature_MGP, axis=1)

            feature_Set_list.append(tmp_feature_Set)
            feature_MGP_list.append(tmp_feature_MGP)

            start += bin

        feature_Set_list = tf.concat(feature_Set_list, axis=1)
        feature_MGP_list = tf.concat(feature_MGP_list, axis=1)

        feature_output = tf.concat(
            [feature_MGP_list, feature_Set_list], axis=1)

        return feature_output

    def model(self, inputsize: tuple):
        input = tf.keras.Input(
            shape=inputsize, name='input_layer'
        )
        model = tf.keras.models.Model(
            inputs=input, outputs=self.call(input)
        )
        plot_model(model, to_file='model.png', show_shapes=True)
        return model
