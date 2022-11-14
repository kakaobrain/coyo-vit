import tensorflow as tf

from . import augment
from . import autoaugment
from . import efficientnet_v1
from . import efficientnet_v2
from . import inception
from . import resnet


class Preprocessor(object):
    def __init__(self, image_size, image_dtype, is_training):
        self.image_size = image_size
        self.image_dtype = image_dtype
        self.is_training = is_training

    def preprocess_image(self, image, preprocess_config):
        is_raw = (image.dtype == tf.string)
        image = tf.image.decode_image(image, channels=3) if is_raw else image
        image.set_shape([None, None, 3])

        for conf in preprocess_config:
            image = getattr(self, conf['type'])(image, conf)

        return tf.cast(image, dtype=self.image_dtype)

    def preprocess_label(self, features, num_classes, supervised_key='label', normalize=False):

        label = features[supervised_key]

        if 'labels' in supervised_key:
            label = tf.math.reduce_max(tf.one_hot(label, num_classes), axis=0)
        else:
            label = tf.one_hot(label, num_classes)

        if normalize is True:
            labels_sum = tf.reduce_sum(label)
            label = tf.where(tf.greater(labels_sum, 0), label / labels_sum, label)
        return label

    def make_cutmix_mask(self, cutmix_alpha, h, w):
        return augment.cutmix_mask(cutmix_alpha, h, w)

    def make_mix(self, batch_size, mixup_alpha, cutmix_alpha, features, labels):
        return augment.mixing(batch_size, mixup_alpha, cutmix_alpha, features, labels)

    def efficientnetv2_noaug(self, image, conf):
        return efficientnet_v2.preprocess_image(
            image=image,
            image_size=self.image_size,
            is_training=self.is_training,
            image_dtype=self.image_dtype,
        )

    def efficientnetv2_finetune(self, image, conf):
        return efficientnet_v2.preprocess_image(
            image=image,
            image_size=self.image_size,
            is_training=self.is_training,
            image_dtype=self.image_dtype,
            augname='ft',
        )

    def efficientnetv2_eval(self, image, conf):
        assert self.is_training is False
        return efficientnet_v2.preprocess_image(
            image=image,
            image_size=self.image_size,
            is_training=self.is_training,
            image_dtype=self.image_dtype,
        )

    def efficientnetv2(self, image, conf):
        return efficientnet_v2.preprocess_image(
            image=image,
            image_size=self.image_size,
            is_training=self.is_training,
            image_dtype=self.image_dtype,
            **conf['params'])

    def effnetv1_autoaugment(self, image, conf):
        return self.efficientnetv2(image, conf)

    def effnetv1_randaugment(self, image, conf):
        return self.efficientnetv2(image, conf)

    def autoaug(self, image, conf):
        input_image_type = image.dtype
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, dtype=tf.uint8)
        image = autoaugment.distort_image_with_autoaugment(image, **conf.get("params",
                                                                             {}))  # augmentation_name: 'v0' or 'test'
        image = tf.cast(image, dtype=input_image_type)
        return image

    def randaug(self, image, conf):
        input_image_type = image.dtype
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, dtype=tf.uint8)
        image = autoaugment.distort_image_with_randaugment(image, **conf.get("params", {}))  # num_layers, magnitude
        image = tf.cast(image, dtype=input_image_type)
        return image

    def resize(self, image, conf):
        # TODO; if is_training, make 4 interpolation variations
        return tf.image.resize(image, list(conf['params']['size']))

    def inception_random_crop(self, image, conf):
        return augment.distorted_bounding_box_crop(image)

    def central_crop(self, image, conf):
        # central_fraction value (0.875) is from inception.py code.
        return tf.image.central_crop(image, central_fraction=conf['params']['central_fraction'])

    def inception_center_crop(self, image, conf):
        # central_fraction value (0.875) is from inception.py code.
        return augment.inception_center_crop(image, size=conf['params']['size'])

    def static_random_crop(self, image, conf):
        return tf.image.random_crop(image, size=list(conf['params']['size']))

    def padded_center_crop(self, image, conf):
        return augment.padded_center_crop(image, **conf['params'])

    def tf_central_crop(self, image, conf):
        return tf.image.central_crop(image, **conf['params'])

    def random_hflip(self, image, conf):
        return tf.image.random_flip_left_right(image)

    def normalize(self, image, conf):
        image -= tf.constant(conf['params']['mean'], shape=(1, 1, 3), dtype=tf.float32)
        image /= tf.constant(conf['params']['std'], shape=(1, 1, 3), dtype=tf.float32)
        return image

    def vit_inception_crop(self, image, conf):
        return augment.vit_inception_crop(image, size=conf['params']['size'])

    def resize_smaller_aspect_ratio(self, image, conf):
        return augment.resize_smaller_aspect_ratio(image, size=conf['params']['size'])

    def InceptionCrop(self, image, conf):
        return augment.InceptionCrop(image, size=conf['params']['size'])

    def ResizeSmall(self, image, conf):
        return augment.ResizeSmall(image, size=conf['params']['size'])

    def CentralCrop(self, image, conf):
        return augment.CentralCrop(image, size=conf['params']['size'])
