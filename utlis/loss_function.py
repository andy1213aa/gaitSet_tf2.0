import tensorflow as tf
from tensorflow_triplet_loss.model import triplet_loss
import tensorflow_addons as tfa

def GaitSet_loss(imgs_embedding,subject):

    
    indices = tf.range(start=0, limit=tf.shape(imgs_embedding)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_embedding = tf.gather(imgs_embedding, shuffled_indices)
    shuffled_subject = tf.gather(subject, shuffled_indices)

    TripletSemiHardLoss = tfa.losses.TripletSemiHardLoss(margin=1.0)
    loss = TripletSemiHardLoss(shuffled_subject, shuffled_embedding)
    # loss = triplet_loss.batch_all_triplet_loss(subject, imgs_embedding, margin=0.8)[0]
    return loss