import numpy as np
import tensorflow as tf
from model.gaitset import GaitSet
import utlis.loss_function as utlis_loss
from utlis.create_training_data import create_training_data
from utlis.save_model import Save_Model
from utlis.config.data_info import OU_MVLP_GaitSet


def main():

    def reduce_dict(d: dict):
        """ inplace reduction of items in dictionary d """
        return {
            k: data_loader.strategy.reduce(
                tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in d.items()
        }

    @tf.function
    def distributed_train_step(data, label):
        results = data_loader.strategy.run(train_step, args=(data, label))
        results = reduce_dict(results)
        return results

    def train_step(data, label):
 
        data = tf.reshape(
            data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4]))

        label = tf.reshape(label, (label.shape[0]*label.shape[1], ))
        result = {}

        with tf.GradientTape() as tape:

            embedding = gaitset(data, training=True)
            embedding = tf.reshape(embedding, (embedding.shape[0], embedding.shape[1] * embedding.shape[2]))
            triplet_loss = utlis_loss.GaitSet_loss(embedding, label)

        result.update({
            'gaitset_loss': triplet_loss
        })

        gaitset_gradient = tape.gradient(
            triplet_loss, gaitset.trainable_variables)

        gaitset_optimizer.apply_gradients(
            zip(gaitset_gradient, gaitset.trainable_variables))

        return result

    def combineImages(images, col, row):
        images = (images+1)/2
        images = images.numpy()
        b, h, w, _ = images.shape
        imagesCombine = np.zeros(shape=(h*col, w*row, 3))
        for y in range(col):
            for x in range(row):
                imagesCombine[y*h:(y+1)*h, x*w:(x+1)*w] = images[x+y*row]
        return imagesCombine

    data_loader = create_training_data('OU_MVLP_GaitSet')
    training_batch = data_loader.read()

    with data_loader.strategy.scope():

        gaitset = GaitSet().model((

            # OU_MVLP_GaitSet['resolution']['k'],
            OU_MVLP_GaitSet['resolution']['height'],
            OU_MVLP_GaitSet['resolution']['width'],
            OU_MVLP_GaitSet['resolution']['channel']
        ))

        gaitset_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4, decay=0)

        gaitset.compile(optimizer=gaitset_optimizer)

    models = {
        'gaitset': gaitset,
    }
    # gaitset.summary()

    log_path = OU_MVLP_GaitSet['save_model']['logdir']
    save_model = Save_Model(models, info=OU_MVLP_GaitSet)
    summary_writer = tf.summary.create_file_writer(
        f'{log_path}/{save_model.startingDate}')

    iteration = 0
    while iteration < 100:
        for step, batch in enumerate(training_batch):

            imgs, subjects = batch

            result = distributed_train_step(imgs, subjects)
            output_message = ''

            with summary_writer.as_default():
                
                for loss_name, loss in result.items():

                    tf.summary.scalar(loss_name, loss,
                                      gaitset_optimizer.iterations)
                    output_message += f'{loss_name}: {loss: .5f}, '

        print(f'Epoch: {iteration:6} Step: {step} {output_message}')

        iteration += 1
        
        # if iteration % 1 == 0:
        #     source_img = batch[0].values[0]
        #     target_img = batch[1].values[0]
        #     source_angle = batch[2].values[0]
        #     target_angle = batch[3].values[0]

        #     # source_img, target_img, source_angle, target_angle = batch
        #     fake_target_img = gaitset([source_img, target_angle])
        #     col_num = 4
        #     row_num = OU_MVLP_gaitset_train['training_info']['batch_size'] // col_num
        #     rawImage = combineImages(target_img, col=col_num, row=row_num)
        #     fakeImage = combineImages(
        #         fake_target_img, col=col_num, row=row_num)

        #     with summary_writer.as_default():
        #         tf.summary.image('rawImage', [rawImage], step=iteration)
        #         tf.summary.image('fakeImage', [fakeImage], step=iteration)
        save_model.save()


if __name__ == '__main__':
    main()
