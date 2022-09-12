import tensorflow as tf
from tensorflow_triplet_loss.model import triplet_loss


def GaitSet_loss(imgs_embedding,subject):
    loss = triplet_loss.batch_all_triplet_loss(subject, imgs_embedding, margin=0.3)[0]
    return loss