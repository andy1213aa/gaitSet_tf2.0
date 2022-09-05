import numpy as np
import tensorflow as tf
from utlis.config import data_info
import utlis.data_loader  

def create_training_data(dataset):

    #import necessary info and class
    dataset_info = getattr(data_info, f'{dataset}_train')
    data_loader_cls = getattr(utlis.data_loader, dataset)

    #create loader and read(with parse) dataset
    data_loader = data_loader_cls(dataset_info)
    # training_batch = data_loader.read()

    return data_loader
    







    
