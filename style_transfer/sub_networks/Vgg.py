from style_transfer.sub_networks.Sub_network import Sub_network
import tensorflow as tf
from style_transfer.layer import upsample_nearest


class VGG(Sub_network):
    _VGG19 = [
        ('prep', 'prep', {}),
        ('conv', 'conv1_1', {'filters': 64}),
        ('conv', 'conv1_2', {'filters': 64}),
        ('pool', 'pool1', {}),
        ('conv', 'conv2_1', {'filters': 128}),
        ('conv', 'conv2_2', {'filters': 128}),
        ('pool', 'pool2', {}),
        ('conv', 'conv3_1', {'filters': 256}),
        ('conv', 'conv3_2', {'filters': 256}),
        ('conv', 'conv3_3', {'filters': 256}),
        ('conv', 'conv3_4', {'filters': 256}),
        ('pool', 'pool3', {}),
        ('conv', 'conv4_1', {'filters': 512}),
        ('conv', 'conv4_2', {'filters': 512}),
        ('conv', 'conv4_3', {'filters': 512}),
        ('conv', 'conv4_4', {'filters': 512}),
        ('pool', 'pool4', {}),
        ('conv', 'conv5_1', {'filters': 512}),
        ('conv', 'conv5_2', {'filters': 512}),
        ('conv', 'conv5_3', {'filters': 512}),
        ('conv', 'conv5_4', {'filters': 512}),
        ('pool', 'pool5', {})
    ]

    _DECODER = [
        ('conv', 'conv4_1', {'filters': 256}),
        ('upsample', 'upsample3', {}),
        ('conv', 'conv3_4', {'filters': 256}),
        ('conv', 'conv3_3', {'filters': 256}),
        ('conv', 'conv3_2', {'filters': 256}),
        ('conv', 'conv3_1', {'filters': 128}),
        ('upsample', 'upsample2', {}),
        ('conv', 'conv2_2', {'filters': 128}),
        ('conv', 'conv2_1', {'filters': 64}),
        ('upsample', 'upsample1', {}),
        ('conv', 'conv1_2', {'filters': 64}),
        ('conv', 'conv1_1', {'filters': 3})
    ]

    def build_subnetwork(self, inputs, weights,
                  last_layer='conv4_1'
                  ):
        definition = self._truncate(self._VGG19, [last_layer])
        with tf.compat.v1.variable_scope('vgg'):
            layers = self._build_net(definition, inputs, weights,
                                activation=tf.nn.relu, trainable=False)
        return layers

    def subnetwork_layer_params(self, layer):
        for _, name, params in self._VGG19:
            if name == layer:
                return params
        raise ValueError('Unknown layer: ' + layer)

    def build_decoder(self, inputs, weights, trainable,
                      activation=tf.nn.relu):
        with tf.compat.v1.variable_scope('decoder'):
            layers = self._build_net(self._DECODER, inputs, weights,
                                activation=activation, trainable=trainable)
            return layers['conv1_1']

    def _build_net(self, definition, inputs, weights, activation, trainable):
        layer, layers = inputs, {}
        for type, name, params in definition:
            if type == 'conv':
                
                layer = tf.pad(tensor=layer, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
                                   mode='reflect')
                if weights:  # pretrained weights provided
                    W_init = tf.compat.v1.constant_initializer(weights[name + '_W'])
                    b_init = tf.compat.v1.constant_initializer(weights[name + '_b'])
                else:
                    W_init = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
                    b_init = tf.compat.v1.zeros_initializer()
                layer = tf.compat.v1.layers.conv2d(layer,
                                         name=name,
                                         padding='valid',
                                         activation=activation,
                                         kernel_size=3,
                                         kernel_initializer=W_init,
                                         bias_initializer=b_init,
                                         trainable=trainable,                                       
                                         **params)
            elif type == 'pool':
                layer = tf.compat.v1.layers.max_pooling2d(layer,
                                                name=name, strides=2, pool_size=2
                                                )
            elif type == 'upsample':
                layer = upsample_nearest(layer, scale=2)
            elif type == 'prep':
                layer = self._vgg_preprocess(layer)
            else:
                raise ValueError('Unknown layer: %s' % type)
            layers[name] = layer
        return layers

    def _truncate(self, definition, used_layers):
        names = [name for _, name, _ in definition]
        return definition[:max(names.index(name) for name in used_layers) + 1]

    def _vgg_preprocess(self, inputs):
        """Preprocess image for the VGG network using the convolutional layer

        The layer expects an RGB image with pixel values in [0,1].
        The layer flips the channels (RGB -> BGR), scales the values to [0,255] range,
        and subtracts the VGG mean pixel.
        """
        #data_format = 'NCHW' if data_format == 'channels_first' else 'NHWC'
        W = tf.Variable([[[
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0]
        ]]], trainable=False, dtype=tf.float32)
        # VGG19 mean pixel value is taken from
        # https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
        b = tf.Variable([-103.939, -116.779, -123.68], trainable=False, dtype=tf.float32)
        conv2d = tf.nn.conv2d(input=inputs, filters=W, strides=(1, 1, 1, 1), padding='VALID')
        return tf.nn.bias_add(conv2d, b)
