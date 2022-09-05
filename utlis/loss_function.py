import tensorflow as tf


def generator_loss(fake_logit):
    loss = -tf.math.reduce_mean(fake_logit)
    # loss = -tf.reduce_mean(fake_logit)
    return loss


def discriminator_loss(real_logit, fake_logit):
    # real_logit_loss = -tf.math.reduce_mean(real_logit)
    # fake_logit_loss = tf.math.reduce_mean(fake_logit)

    real_logit_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
    fake_logit_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))

    return real_logit_loss, fake_logit_loss


def generator_view_classification_loss(fake_view_logit, target_angle):
    # loss = -tf.reduce_mean(fake_view_logit)
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(target_angle, fake_view_logit)
    return loss


def discriminator_view_classification_loss(real_view_logit, source_angle):
    # real_view_loss = -tf.math.reduce_mean(real_view_logit)
    # fake_view_loss = tf.math.reduce_mean(fake_view_logit)
    # real_view_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_view_logit))
    # fake_view_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_view_logit))
    loss = (source_angle, real_view_logit)
    return loss


def cycle_consistency_loss(X, fake):
    loss = tf.math.reduce_mean(tf.abs(X-fake))
    return loss


# def identification_loss(real_logit, fake_logit):
#     loss = tf.math.reduce_mean(tf.math.log(
#         real_logit)) + tf.math.reduce_mean(tf.math.log(1-fake_logit))
#     return loss
