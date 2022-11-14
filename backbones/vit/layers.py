import tensorflow as tf


class Identity(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Identity, self).__init__(name=name)

    def call(self, x):
        return x
