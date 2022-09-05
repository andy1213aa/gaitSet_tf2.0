import numpy as np
import tensorflow as tf
from model.Generator import Generator
from model.Discriminator import Discriminator
# from model.Identification_discriminator import Identification_discriminator
import utlis.loss_function as utlis_loss
from utlis.create_training_data import create_training_data
from utlis.save_model import Save_Model
from utlis.config.data_info import OU_MVLP_multi_view_train, training_info


def main():

    def reduce_dict(d: dict):
        """ inplace reduction of items in dictionary d """
        return {
            k: data_loader.strategy.reduce(
                tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in d.items()
        }

    @tf.function
    def distributed_train_step(source_img, target_img, source_angle, target_angle, source_angle_onehot, target_angle_onehot):
        results = data_loader.strategy.run(train_step, args=(
            source_img, target_img, source_angle, target_angle, source_angle_onehot, target_angle_onehot))
        results = reduce_dict(results)
        return results

    def train_step(source_img, target_img, source_angle, target_angle, source_angle_onehot, target_angle_onehot):

        result = {}
        with tf.GradientTape(persistent=True) as tape:

            '''
            Cycle Consistency Loss
            '''
            fake_target_image = generator(
                [source_img, target_angle], training=True)
            reconstruct_source_image = generator(
                [fake_target_image, source_angle], training=True)
            cycle_loss = utlis_loss.cycle_consistency_loss(
                source_img, reconstruct_source_image)  

            '''
            Adversarial Loss
            '''
            real_logit, predict_label_from_real_logit = discriminator(
                source_img, training=True)

            fake_logit, predict_label_from_fake_logit = discriminator(
                fake_target_image, training=True)

            adversarial_generator_loss = utlis_loss.generator_loss(
                fake_logit)

            real_logit_loss, fake_logit_loss = utlis_loss.discriminator_loss(
                real_logit, fake_logit)

            '''
            View Classification Loss
            '''
            # predict_label_from_real_logit = View_discriminator(target_img)
            # predict_label_from_fake_logit = View_discriminator(
            #     fake_target_image)

            gen_view_loss = tf.math.reduce_mean(view_classification_loss(
                target_angle_onehot, predict_label_from_fake_logit)) 

            dis_view_loss = tf.math.reduce_mean(view_classification_loss(
                source_angle_onehot, predict_label_from_real_logit))

            # utlis_loss.generator_view_classification_loss(
            #     predict_label_from_fake_logit, target_angle_onehot)
            # dis_real_view_loss, dis_fake_view_loss = utlis_loss.discriminator_view_classification_loss(
            #     predict_label_from_real_logit, source_angle_onehot
            # )

            '''
            Total Loss
            '''
            generator_loss = cycle_loss + adversarial_generator_loss + gen_view_loss
            # fake_view_classification_loss
            discriminator_loss = real_logit_loss + fake_logit_loss + dis_view_loss
            # View_discriminator_loss = real_view_classification_loss + \
            #     fake_view_classification_loss

        result.update({
            'G_advloss': adversarial_generator_loss,
            'G_cycle_loss': cycle_loss,
            'G_view_loss': gen_view_loss,
            'D_real_logit_loss': real_logit_loss,
            'D_fake_logit_loss': fake_logit_loss,
            'D_view_loss':dis_view_loss,
            'G_total_loss': generator_loss,
            'D_total_loss': discriminator_loss,
        })

        generator_gradient = tape.gradient(
            generator_loss, generator.trainable_variables)

        discriminator_gradient = tape.gradient(
            discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(generator_gradient, generator.trainable_variables))

        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradient, discriminator.trainable_variables))

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

    data_loader = create_training_data('OU_MVLP_multi_view')
    training_batch = data_loader.read()

    with data_loader.strategy.scope():

        # mode = 'multi_view_gait_generation'
        # date = '2022_7_13_18_14'
        # generator = tf.keras.models.load_model(f'/home/aaron/Desktop/Aaron/College-level_Applied_Research/gait_log/{mode}/{date}/generator/trained_ckpt')
        # discriminator = tf.keras.models.load_model(f'/home/aaron/Desktop/Aaron/College-level_Applied_Research/gait_log/{mode}/{date}/discriminator/trained_ckpt')

        generator = Generator(32).model(
            (OU_MVLP_multi_view_train['resolution']['height'],
             OU_MVLP_multi_view_train['resolution']['width'],
             OU_MVLP_multi_view_train['resolution']['channel']),
            OU_MVLP_multi_view_train['resolution']['angle_nums'])

        discriminator = Discriminator(32).model(
            (OU_MVLP_multi_view_train['resolution']['height'],
             OU_MVLP_multi_view_train['resolution']['width'],
             OU_MVLP_multi_view_train['resolution']['channel']),
            OU_MVLP_multi_view_train['resolution']['angle_nums'])

        # # Identification_discriminator = Identification_discriminator(
        # 32).model((128, 88, 3))

        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=4e-5, decay=0)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=2e-4, decay=0)
        view_classification_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        # generator_optimizer = tf.keras.optimizers.RMSprop(lr=2e-5)
        # discriminator_optimizer = tf.keras.optimizers.RMSprop(
        #     lr=8e-5)
        # Identification_discriminator_optimizer = tf.keras.optimizers.Adam(
        # lr=1e-4, decay=1e-4)

        generator.compile(optimizer=generator_optimizer)
        discriminator.compile(optimizer=discriminator_optimizer)
        # Identification_discriminator.compile(
        # optimizer=Identification_discriminator_optimizer)

    models = {
        'generator': generator,
        'discriminator': discriminator,
        # 'View_discriminator': View_discriminator,
        # # 'Identification_discriminator': Identification_discriminator
    }
    generator.summary()
    discriminator.summary()

    log_path = training_info['save_model']['logdir']
    save_model = Save_Model(models, info=training_info)
    summary_writer = tf.summary.create_file_writer(
        f'{log_path}/{save_model.startingDate}')
    iteration = 0

    while iteration < 45:
        for step, batch in enumerate(training_batch):
            # tf.summary.trace_on(graph=True, profiler=True)
            source_img, target_img, source_angle, target_angle, source_angle_onehot, target_angle_onehot = batch
            result = distributed_train_step(
                source_img, target_img, source_angle, target_angle, source_angle_onehot, target_angle_onehot)
            output_message = ''

            with summary_writer.as_default():

                # tf.summary.trace_export(
                #     name="model_trace",
                #     step=0,
                #     profiler_outdir= f'{log_path}/{save_model.startingDate}')

                for loss_name, loss in result.items():

                    tf.summary.scalar(loss_name, loss,
                                      generator_optimizer.iterations)
                    output_message += f'{loss_name}: {loss: .5f}, '

            print(f'Epoch: {iteration:6} Step: {step} {output_message}')

        iteration += 1

        if iteration % 1 == 0:
            source_img = batch[0].values[0]
            target_img = batch[1].values[0]
            source_angle = batch[2].values[0]
            target_angle = batch[3].values[0]

            # source_img, target_img, source_angle, target_angle = batch
            fake_target_img = generator([source_img, target_angle])
            col_num = 4
            row_num = OU_MVLP_multi_view_train['training_info']['batch_size'] // col_num
            rawImage = combineImages(target_img, col=col_num, row=row_num)
            fakeImage = combineImages(fake_target_img, col=col_num, row=row_num)

            with summary_writer.as_default():
                tf.summary.image('rawImage', [rawImage], step=iteration)
                tf.summary.image('fakeImage', [fakeImage], step=iteration)
        save_model.save()


if __name__ == '__main__':
    main()
