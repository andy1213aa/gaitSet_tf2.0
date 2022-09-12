import tensorflow as tf

training_info = {
    'save_model': {
        'logdir': '/home/aaron/Desktop/Aaron/College-level_Applied_Research/gait_log/multi_view_gait_generation'
    }
}


casia_B_train = {
    "feature": {
        "angle": tf.float32,
        "subject": tf.float32,
        "data_row": tf.string
    },

    "resolution": {
        "height": 64,
        "width": 64,
        "channel": None
    },

    "training_info": {
        "size": 73,
        "batch_size": 64,
        "shuffle": True
    }

}

OU_MVLP_triplet_train = {

    "feature": {
        "imgs": tf.string,
        "subject": tf.string,
        "angles": tf.string

    },

    "resolution": {
        "height": 128,
        "width": 88,
        "channel": 3,
        "angle_nums": 14,
        "k": 4
    },

    "training_info": {
        "tfrecord_path": '/home/aaron/Desktop/Aaron/College-level_Applied_Research/tfrecord/OUMVLP_Triplet/triplet_train_4inPerson.tfrecords',
        "data_num": 50000,
        "batch_size": 16,
        "shuffle": True
    }
}

OU_MVLP_multi_view_train = {

    "feature": {
        "img1": tf.string,
        "img2": tf.string,
        "angle1": tf.string,
        "angle2": tf.string,
        "subject": tf.string,
        "angle1_onehot": tf.string,
        "angle2_onehot": tf.string,

    },

    "resolution": {
        "height": 128,
        "width": 88,
        "channel": 1,
        "angle_nums": 14,
    },

    "training_info": {
        "tfrecord_path": '/home/aaron/Desktop/Aaron/College-level_Applied_Research/tfrecord/OUMVLP_multiview/multi_view_gait_train_5154.tfrecords',
        "data_num": 25000,
        "batch_size": 32,
        "shuffle": True
    }
}

OU_MVLP_GaitSet = {

    "feature": {
        "imgs": tf.string,
        "subject": tf.string,
        "angles": tf.string

    },

    "resolution": {
        "height": 64,
        "width": 64,
        "channel": 1,
        "k": 16,
        "angle_nums": 14,
    },

    "training_info": {
        "tfrecord_path": '/media/aaron/新增磁碟區/ITRI_SSTC/S100/gait/tf_record/gaitset_10k_64x64_16p.tfrecords',
        "data_num": 1000,
        "batch_size": 8,
        "shuffle": True
    },

    'save_model': {
        'logdir': '/home/aaron/Desktop/Aaron/S100/College-level_Applied_Research/gait_log/gaitset'
    }
}