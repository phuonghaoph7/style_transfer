import numpy as np
import tensorflow as tf


def upsample_nearest(inputs, scale):
    shape = tf.shape(input=inputs)
    n, h, w, c = shape[0], shape[1], shape[2], shape[3]
    return tf.image.resize(inputs, tf.stack([h*scale, w*scale]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)