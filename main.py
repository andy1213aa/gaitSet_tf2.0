import re
from jsonschema import RefResolutionError
import numpy as np
import tensorflow as tf
from model.gaitset import GaitSet
import utlis.loss_function as utlis_loss
from utlis.create_training_data import create_training_data
from utlis.save_model import Save_Model
from utlis.config.data_info import OU_MVLP_GaitSet
from einops import rearrange, reduce, repeat
from tqdm import tqdm




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

        data = rearrange(data, 'b p s h w c -> (b p) s h w c')
        label = rearrange(label, 'b p -> (b p)')
        # label = tf.reshape(label, (label.shape[0]*label.shape[1], ))
        
        result = {}

        with tf.GradientTape() as tape:
        
            embedding = gaitset(data, training=True)
            embedding = rearrange(embedding, 'b h c -> b (h c)')
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
    training_batch, vali_data_batch = data_loader.read()

    with data_loader.strategy.scope():

        gaitset = GaitSet(256).model((

            OU_MVLP_GaitSet['resolution']['k'],
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
    gaitset.summary()

    log_path = OU_MVLP_GaitSet['save_model']['logdir']
    save_model = Save_Model(models, info=OU_MVLP_GaitSet)
    summary_writer = tf.summary.create_file_writer(
        f'{log_path}/{save_model.startingDate}')

    iteration = 0
    while iteration < 20:
        for step, batch in enumerate(tqdm(training_batch)):

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
        
        if iteration % 1 == 0:

            prob = []
            gallery = []

            prob_label_clt = []
            gallery_label_clt = []
            for step, batch in enumerate(tqdm(vali_data_batch)):
                validate_imgs, validate_subjects = batch
                
                prob_imgs = validate_imgs[:, 0:2, :, :, :, :]
                gallery_imgs = validate_imgs[:, 2:, :, :, :, :]

                # validate_imgs = rearrange(validate_imgs, 'b p s h w c -> (b p) s h w c')
                
                prob_imgs = rearrange(prob_imgs, 'b p s h w c -> (b p) s h w c')
                gallery_imgs = rearrange(gallery_imgs, 'b p s h w c -> (b p) s h w c')
                
                prob_feature= gaitset(prob_imgs)
                gallery_feature= gaitset(gallery_imgs)


                prob_feature = rearrange(prob_feature, 'b h c -> b (h c)')
                gallery_feature = rearrange(gallery_feature, 'b h c -> b (h c)')
           
                
                prob.append(prob_feature)
                gallery.append(gallery_feature)

                prob_label = validate_subjects[:, 0:2]
                gallery_label = validate_subjects[:, 2:]
                
                prob_label = rearrange(prob_label, 'b p -> (b p)')
                gallery_label = rearrange(gallery_label, 'b p -> (b p)')

                prob_label_clt.append(prob_label)
                gallery_label_clt.append(gallery_label)
                
            
            prob = np.concatenate( prob, axis=0 )
            gallery = np.concatenate( gallery, axis=0 )

            prob_label_clt = np.concatenate( prob_label_clt, axis=0 )
            gallery_label_clt = np.concatenate( gallery_label_clt, axis=0 )
            
            print(prob.shape)
            print(gallery.shape)
            print(prob_label_clt.shape)
            print(gallery_label_clt.shape)
            # RANK1
            cos_correct_cnt = 0
            mse_correct_cnt = 0

            cosine = tf.keras.losses.CosineSimilarity(axis=1,
                        reduction=tf.keras.losses.Reduction.NONE)
            mse = tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.NONE)

            for i, p in enumerate(tqdm(prob)):
                p = np.tile(p, prob.shape[0]).reshape(prob.shape[0], -1)
                
                #cosine
                cosine_loss = cosine(gallery, p).numpy()
                cosine_loss_predict_label = gallery_label_clt[np.argmin(cosine_loss)]

                #Ecludiense
                mse_loss = mse(gallery, p).numpy()
                mse_predict_label = gallery_label_clt[np.argmin(mse_loss)]

                if cosine_loss_predict_label == prob_label_clt[i]:
                    cos_correct_cnt+=1
                if mse_predict_label == prob_label_clt[i]:
                    mse_correct_cnt+=1

            print(f'Rank1: \n \
            cos:{(cos_correct_cnt / prob.shape[0]) * 100}%, mse: {(mse_correct_cnt / prob.shape[0]) * 100}%')


        #     for embedding in embedding_list:

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
