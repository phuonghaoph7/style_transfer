import tensorflow as tf
import numpy as np
from .configs import config

from tensorflow.python.layers import utils
def adain(content, style, epsilon=1e-5):
    axes =  [1,2]

    c_mean, c_var = tf.nn.moments(x=content, axes=axes, keepdims=True)
    s_mean, s_var = tf.nn.moments(x=style, axes=axes, keepdims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    return s_std * (content - c_mean) / c_std + s_mean
'''def expand_moments_dim(moment):
    return tf.reshape(moment, [-1, 1, 1, tf.shape(moment)[-1]])

def adain(content_feature, style_feature, eps=1e-5):
    content_mean, content_var = tf.nn.moments(content_feature, axes=[1, 2])
    style_mean, style_var = tf.nn.moments(style_feature, axes=[1, 2])

    content_std = tf.sqrt(content_var)
    style_std = tf.sqrt(style_var)

    content_mean = expand_moments_dim(content_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_mean = tf.broadcast_to(content_mean, tf.shape(content_feature))

    content_std = expand_moments_dim(content_std) + eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_std = tf.broadcast_to(content_std, tf.shape(content_feature))

    style_mean = expand_moments_dim(style_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_mean = tf.broadcast_to(style_mean, tf.shape(content_feature))

    style_std = expand_moments_dim(style_std) + eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_std = tf.broadcast_to(style_std, tf.shape(content_feature))

    normalized_content = tf.divide(content_feature - content_mean, content_std)
    return tf.multiply(normalized_content, style_std) + style_mean'''

