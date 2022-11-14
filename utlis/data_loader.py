
import numpy as np
import tensorflow as tf
from einops import rearrange, reduce, repeat

class GenericTFLoader():
    '''
    load tfrecord data.
    '''

    def __init__(self, config):
        self._config = config

    def read(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError

    # @classmethod
    # def generate_loader(cls, loader_subclasses, config):
    #     loader_collection = []
    #     for loader in loader_subclasses:
    #         loader_subclass()


class OU_MVLP_multi_view(GenericTFLoader):

    def __init__(self, config):
        self._config = config
        self.strategy = tf.distribute.MirroredStrategy(
            devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    def read(self):

        # per replica batch size
        BATCH_SIZE = self._config['training_info']['batch_size']

        # initialize tf.distribute.MirroredStrategy

        GLOBAL_BATCH_SIZE = self.strategy.num_replicas_in_sync * BATCH_SIZE

        print(f'Number of devices: {self.strategy.num_replicas_in_sync}')

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_set = tf.data.TFRecordDataset(
            self._config['training_info']['tfrecord_path'])
        data_set = data_set.map(self.parse, num_parallel_calls=AUTOTUNE)
        # data_set = data_set.cache()
        train_data = data_set.take(self._config['training_info']['train_size'])
        vali_data = train_data.take(self._config['training_info']['validate_size'])
        
        train_data = train_data.shuffle(
            100, reshuffle_each_iteration=self._config['training_info']['shuffle'])
        train_data = train_data.batch(
            GLOBAL_BATCH_SIZE, drop_remainder=True)

        vali_data = vali_data.batch(
            self._config['training_info']['vali_batch_size'], drop_remainder=True)

        train_data = train_data.prefetch(buffer_size=AUTOTUNE)
        train_data_ds = self.strategy.experimental_distribute_dataset(
            train_data)

        return train_data_ds, vali_data

    def parse(self, example_proto):

        features = tf.io.parse_single_example(
            example_proto,
            features={key: tf.io.FixedLenFeature(
                [], self._config['feature'][key]) for key in self._config['feature']}

        )
        img1 = features['img1']
        img2 = features['img2']

        angle1 = features['angle1']
        angle2 = features['angle2']

        angle1_onehot = features['angle1_onehot']
        angle2_onehot = features['angle2_onehot']

        # subject = features['subject']

        img1 = tf.io.decode_raw(img1, np.float32)
        img2 = tf.io.decode_raw(img2, np.float32)

        angle1 = tf.io.decode_raw(angle1, np.float32)
        angle2 = tf.io.decode_raw(angle2, np.float32)

        angle1_onehot = tf.io.decode_raw(angle1_onehot, np.float32)
        angle2_onehot = tf.io.decode_raw(angle2_onehot, np.float32)
        # subject = tf.io.decode_raw(subject, np.float32)

        img1 = tf.reshape(img1, (self._config['resolution']['height'],
                          self._config['resolution']['width'], self._config['resolution']['channel']))
        img2 = tf.reshape(img2, (self._config['resolution']['height'],
                          self._config['resolution']['width'], self._config['resolution']['channel']))

        img1 = (img1 - 127.5) / 127.5
        img2 = (img2 - 127.5) / 127.5

        angle1 = tf.reshape(angle1, (self._config['resolution']['height'],
                            self._config['resolution']['width'], self._config['resolution']['angle_nums']))
        angle2 = tf.reshape(angle2, (self._config['resolution']['height'],
                            self._config['resolution']['width'], self._config['resolution']['angle_nums']))

        angle1_onehot = tf.reshape(
            angle1_onehot, (self._config['resolution']['angle_nums'],))
        angle2_onehot = tf.reshape(
            angle2_onehot, (self._config['resolution']['angle_nums'],))
        # subject = tf.reshape(subject, (1,))

        return [img1, img2, angle1, angle2, angle1_onehot, angle2_onehot]


class OU_MVLP_GaitSet(GenericTFLoader):

    def __init__(self, config):
        self._config = config
        self.strategy = tf.distribute.MirroredStrategy(
            devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    def read(self):

        # per replica batch size
        BATCH_SIZE = self._config['training_info']['batch_size']

        # initialize tf.distribute.MirroredStrategy

        GLOBAL_BATCH_SIZE = self.strategy.num_replicas_in_sync * BATCH_SIZE

        print(f'Number of devices: {self.strategy.num_replicas_in_sync}')

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_set = tf.data.TFRecordDataset(
            self._config['training_info']['tfrecord_path'])
        data_set = data_set.map(self.parse, num_parallel_calls=AUTOTUNE)
        # data_set = data_set.cache()
        train_data = data_set.take(self._config['training_info']['train_size'])
        vali_data = train_data.take(self._config['training_info']['validate_size'])
        
        train_data = train_data.shuffle(
            5000, reshuffle_each_iteration=self._config['training_info']['shuffle'])
        train_data = train_data.batch(
            GLOBAL_BATCH_SIZE, drop_remainder=True)

        vali_data = vali_data.shuffle(
            1000, reshuffle_each_iteration=self._config['training_info']['shuffle'])
        vali_data = vali_data.batch(
            self._config['training_info']['vali_batch_size'], drop_remainder=True)

        train_data = train_data.prefetch(buffer_size=AUTOTUNE)
        vali_data = vali_data.prefetch(buffer_size=AUTOTUNE)

        train_data_ds = self.strategy.experimental_distribute_dataset(
            train_data)

        return train_data_ds, vali_data

    def parse(self, example_proto):

        features = tf.io.parse_single_example(
            example_proto,
            features={key: tf.io.FixedLenFeature(
                [], self._config['feature'][key]) for key in self._config['feature']}

        )

        imgs = features['imgs']
        subject = features['subject']
   

        imgs = tf.io.decode_raw(imgs, tf.float32)
        imgs = tf.reshape(imgs,  (self._config['resolution']['p'], self._config['resolution']['k'], self._config['resolution']['height'], 
                                self._config['resolution']['width'], self._config['resolution']['channel']))
        imgs = imgs/255.

        subject = tf.io.decode_raw(subject, tf.float32)
        subject = tf.reshape(subject, (self._config['resolution']['p'],))

        return [imgs, subject]
