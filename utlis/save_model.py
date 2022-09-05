import tensorflow as tf
import numpy as np
import datetime
import os
from shutil import copytree, copyfile
from pathlib import Path
from tensorflow.keras.utils import plot_model
# Dont show warning about:
# WARNING:absl:Found untraced functions such as conv1_1_layer_call_fn, conv1_1_layer_call_and_return_conditional_losses, _jit_compiled_convolution_op,
# leakyReLU1_1_layer_call_fn, leakyReLU1_1_layer_call_and_return_conditional_losses
# while saving (showing 5 of 60). These functions will not be directly callable after loading.
# >>>>>>
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# <<<<<<


class Save_Model(tf.keras.callbacks.Callback):
    def __init__(self, models: dict, info, mode='min', save_weights_only=False):
        super(Save_Model, self).__init__()

        '''
        Input:
            models: {model name: model object}
            (ex. {'Generator': some keras model}
        mode: 
            not use for now.
        '''
        self.models = models
        log_path = info['save_model']['logdir']
        # setting directory of saving weight
        # self.dataSetConfig = dataSetConfig

        # biggest better or lowest better
        self.mode = mode
        # save type
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf

        self.counter = 0
        self.training = True
        self.epoch = 1

        startingTime = datetime.datetime.now()
        startingDate = f'{startingTime.year}_{startingTime.month}_{startingTime.day}' + \
            '_'+f'{startingTime.hour}_{startingTime.minute}'
        self.startingDate = startingDate  # log starting date
        os.mkdir(f"{log_path}/{startingDate}")

        self.modelDir = {}
        for model_name, model in models.items():
            self.modelDir[model_name] = f"{log_path}/{startingDate}/{model_name}/"
           
            if not os.path.isdir(f"{log_path}/{startingDate}/{model_name}/"):
                os.mkdir(f"{log_path}/{startingDate}/{model_name}/")
                plot_model(model, to_file=f"{log_path}/{startingDate}/{model_name}/{model_name}.png", show_shapes=True)

        work_dir = os.path.abspath('')
        copytree(f'{work_dir}/model',
                 f'{log_path}/{startingDate}/model')
        copytree(f'{work_dir}/utlis',
                 f'{log_path}/{startingDate}/utlis')
        copyfile(
            f'{work_dir}/main.py', f'{log_path}/{startingDate}/main.py')

    def save(self):
        if self.save_weights_only:
            for model_name, model in self.models.items():
                model.save_weights(self.modelDir[model_name] + "trained_ckpt")
        else:
            for model_name, model in self.models.items():
                model.save(self.modelDir[model_name] + "trained_ckpt")

    # def save_config(self, monitor_value):
    #     saveLogTxt = f"""
    # Parameter Setting
    # =======================================================
    # DataSet: { self.dataSetConfig['dataSet']}
    # DataShape: ({ self.dataSetConfig['length']}, { self.dataSetConfig['width']}, {self.dataSetConfig['height']})
    # DataSize: {self.dataSetConfig['datasize']}
    # TrainingSize: { self.dataSetConfig['trainSize']}
    # TestingSize: { self.dataSetConfig['testSize']}
    # BatchSize: { self.dataSetConfig['batchSize']}
    # =======================================================
    # Training log
    # =======================================================
    # Training start: { self.dataSetConfig['startingTime']}
    # Training stop: {datetime.datetime.now()}
    # Training epoch: {self.epoch}
    # Root Mean Square Error: {monitor_value}%
    # =======================================================
    # """
    #     with open(self.dataSetConfig['logDir']+'config.txt', 'w') as f:
    #         f.write(saveLogTxt)

    def on_epoch_end(self, monitor_value=0, logs=None):
        # read monitor value from logs
        # monitor_value = logs.get(self.monitor)
        # Create the saving rule

        # if self.mode == 'min' and monitor_value < self.best:

        #     self.best = monitor_value
        #     self.counter = 0
        # elif self.mode == 'max' and monitor_value > self.best:

        #     self.best = monitor_value
        #     self.counter = 0
        # else:
        #     self.counter += 1
        #     if self.counter >= self.dataSetConfig['stopConsecutiveEpoch']:
        #         self.save_model()
        #         self.save_config(monitor_value)
        #         self.training = False
        self.epoch += 1
