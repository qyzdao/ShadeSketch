"""
ShadeSketch
https://github.com/qyzdao/ShadeSketch

Learning to Shadow Hand-drawn Sketches
Qingyuan Zheng, Zhuoru Li, Adam W. Bargteil

Copyright (C) 2020 The respective authors and Project HAT. All rights reserved.
Licensed under MIT license.
"""

import tensorflow as tf

# import keras
keras = tf.keras
K = keras.backend
Layer = keras.layers.Layer
Conv2D = keras.layers.Conv2D
InputSpec = keras.layers.InputSpec
image_data_format = K.image_data_format
activations = keras.activations
initializers = keras.initializers
regularizers = keras.regularizers
constraints = keras.constraints


class Composite(Layer):

    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        self.data_format = data_format

        super(Composite, self).__init__(**kwargs)

    def call(self, inputs):
        line_inputs, shade_inputs = inputs

        return line_inputs + (shade_inputs + 1) * 0.25

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class PixelwiseConcat(Layer):

    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        self.data_format = data_format

        super(PixelwiseConcat, self).__init__(**kwargs)

    def call(self, inputs):
        pixel_inputs, unit_inputs = inputs

        if self.data_format == 'channels_first':
            repeated_unit_inputs = tf.tile(
                K.expand_dims(K.expand_dims(unit_inputs, 2), 2),
                [1, K.shape(pixel_inputs)[2], K.shape(pixel_inputs)[3], 1]
            )
        elif self.data_format == 'channels_last':
            repeated_unit_inputs = tf.tile(
                K.expand_dims(K.expand_dims(unit_inputs, 1), 1),
                [1, K.shape(pixel_inputs)[1], K.shape(pixel_inputs)[2], 1]
            )

        return K.concatenate([pixel_inputs, repeated_unit_inputs])

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0][0], input_shape[0][1] + input_shape[1][1], input_shape[0][2], input_shape[0][3])
        elif self.data_format == 'channels_last':
            return (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3] + input_shape[1][1])


class SubPixelConv2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='same',
                 data_format=None,
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SubPixelConv2D, self).__init__(
            filters=r * r * filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

        if hasattr(tf.nn, 'depth_to_space'):
            self.depth_to_space = tf.nn.depth_to_space
        else:
            self.depth_to_space = tf.depth_to_space

    def phase_shift(self, I):
        if self.data_format == 'channels_first':
            return self.depth_to_space(I, self.r, data_format="NCHW")
        elif self.data_format == 'channels_last':
            return self.depth_to_space(I, self.r, data_format="NHWC")

    def call(self, inputs):
        return self.phase_shift(super(SubPixelConv2D, self).call(inputs))

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            n, c, h, w = super(SubPixelConv2D, self).compute_output_shape(input_shape)
        elif self.data_format == 'channels_last':
            n, h, w, c = super(SubPixelConv2D, self).compute_output_shape(input_shape)

        if h is not None:
            h = int(self.r * h)
        if w is not None:
            w = int(self.r * w)

        c = int(c / (self.r * self.r))

        if self.data_format == 'channels_first':
            return (n, c, h, w)
        elif self.data_format == 'channels_last':
            return (n, h, w, c)

    def get_config(self):
        config = super(Conv2D, self).get_config()

        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] /= self.r * self.r
        config['r'] = self.r

        return config


class SelfAttention(Layer):

    def __init__(self,
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.data_format = data_format
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        kernel_size = (1, 1)
        self.filters = int(input_shape[channel_axis])

        self.kernel_f = self.add_weight(shape=kernel_size + (self.filters, self.filters // 8),
                                        initializer=self.kernel_initializer,
                                        name='kernel_f',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.kernel_g = self.add_weight(shape=kernel_size + (self.filters, self.filters // 8),
                                        initializer=self.kernel_initializer,
                                        name='kernel_g',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.kernel_h = self.add_weight(shape=kernel_size + (self.filters, self.filters),
                                        initializer=self.kernel_initializer,
                                        name='kernel_h',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_f = self.add_weight(shape=(self.filters // 8,),
                                          initializer=self.bias_initializer,
                                          name='bias_f',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)

            self.bias_g = self.add_weight(shape=(self.filters // 8,),
                                          initializer=self.bias_initializer,
                                          name='bias_g',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)

            self.bias_h = self.add_weight(shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          name='bias_h',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)
        else:
            self.bias_f = None
            self.bias_g = None
            self.bias_h = None

        self.gamma = self.add_weight(
            name='gamma',
            shape=(1,),
            initializer=initializers.Constant(0)
        )

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        f = K.conv2d(inputs,
                     self.kernel_f,
                     data_format=self.data_format,
                     strides=(1, 1),
                     dilation_rate=(1, 1))  # [bs, h, w, c']
        g = K.conv2d(inputs,
                     self.kernel_g,
                     data_format=self.data_format,
                     strides=(1, 1),
                     dilation_rate=(1, 1))  # [bs, h, w, c']
        h = K.conv2d(inputs,
                     self.kernel_h,
                     data_format=self.data_format,
                     strides=(1, 1),
                     dilation_rate=(1, 1))  # [bs, h, w, c]

        if self.use_bias:
            f = K.bias_add(f, self.bias_f, data_format=self.data_format)  # [bs, h, w, c']
            g = K.bias_add(g, self.bias_g, data_format=self.data_format)  # [bs, h, w, c']
            h = K.bias_add(h, self.bias_h, data_format=self.data_format)  # [bs, h, w, c]

        # N = h * w
        s = K.dot(K.batch_flatten(g), K.transpose(K.batch_flatten(f)))  # # [bs, N, N]

        beta = K.softmax(s)  # attention map

        o = K.dot(beta, K.batch_flatten(h))  # [bs, N, C]
        o = K.reshape(o, K.shape(inputs))  # [bs, h, w, C]

        return self.activation(self.gamma * o + inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'data_format': self.data_format,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""
Implementation of Coordinate Channel

keras-coordconv

MIT License
Copyright (c) 2018 Somshubra Majumdar

https://github.com/titu1994/keras-coordconv/blob/master/coord.py
"""


class _CoordinateChannel(Layer):
    """ Adds Coordinate Channels to the input tensor.

    # Arguments
        rank: An integer, the rank of the input data-uniform,
            e.g. "2" for 2D convolution.
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        ND tensor with shape:
        `(samples, channels, *)`
        if `data_format` is `"channels_first"`
        or ND tensor with shape:
        `(samples, *, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        ND tensor with shape:
        `(samples, channels + 2, *)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, *, channels + 2)`
        if `data_format` is `"channels_last"`.

    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, rank,
                 use_radius=False,
                 data_format='channels_last',
                 **kwargs):
        super(_CoordinateChannel, self).__init__(**kwargs)

        if data_format not in [None, 'channels_first', 'channels_last']:
            raise ValueError('`data_format` must be either "channels_last", "channels_first" '
                             'or None.')

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = data_format
        self.axis = 1 if image_data_format() == 'channels_first' else -1

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = K.shape(inputs)

        if self.rank == 1:
            input_shape = [input_shape[i] for i in range(3)]
            batch_shape, dim, channels = input_shape

            xx_range = tf.tile(K.expand_dims(K.arange(0, dim), axis=0),
                               K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=-1)

            xx_channels = K.cast(xx_range, K.floatx())
            xx_channels = xx_channels / K.cast(dim - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels], axis=-1)

        if self.rank == 2:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(4)]
            batch_shape, dim1, dim2, channels = input_shape

            xx_ones = tf.ones(K.stack([batch_shape, dim2]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = tf.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                               K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)
            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            yy_ones = tf.ones(K.stack([batch_shape, dim1]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = tf.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                               K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
            yy_channels = (yy_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels, yy_channels], axis=-1)

            if self.use_radius:
                rr = K.sqrt(K.square(xx_channels - 0.5) +
                            K.square(yy_channels - 0.5))
                outputs = K.concatenate([outputs, rr], axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

        if self.rank == 3:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 4, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(5)]
            batch_shape, dim1, dim2, dim3, channels = input_shape

            xx_ones = tf.ones(K.stack([batch_shape, dim3]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = tf.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                               K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)

            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            xx_channels = K.expand_dims(xx_channels, axis=1)
            xx_channels = tf.tile(xx_channels,
                                  [1, dim1, 1, 1, 1])

            yy_ones = tf.ones(K.stack([batch_shape, dim2]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = tf.tile(K.expand_dims(K.arange(0, dim3), axis=0),
                               K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            yy_channels = K.expand_dims(yy_channels, axis=1)
            yy_channels = tf.tile(yy_channels,
                                  [1, dim1, 1, 1, 1])

            zz_range = tf.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                               K.stack([batch_shape, 1]))
            zz_range = K.expand_dims(zz_range, axis=-1)
            zz_range = K.expand_dims(zz_range, axis=-1)

            zz_channels = tf.tile(zz_range,
                                  [1, 1, dim2, dim3])
            zz_channels = K.expand_dims(zz_channels, axis=-1)

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim2 - 1, K.floatx())
            xx_channels = xx_channels * 2 - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim3 - 1, K.floatx())
            yy_channels = yy_channels * 2 - 1.

            zz_channels = K.cast(zz_channels, K.floatx())
            zz_channels = zz_channels / K.cast(dim1 - 1, K.floatx())
            zz_channels = zz_channels * 2 - 1.

            outputs = K.concatenate([inputs, zz_channels, xx_channels, yy_channels],
                                    axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 4, 1, 2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        if self.use_radius and self.rank == 2:
            channel_count = 3
        else:
            channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            'rank': self.rank,
            'use_radius': self.use_radius,
            'data_format': self.data_format
        }
        base_config = super(_CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoordinateChannel1D(_CoordinateChannel):
    """ Adds Coordinate Channels to the input tensor of rank 1.

    # Arguments
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, steps, input_dim + 2)`

    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, data_format=None, **kwargs):
        super(CoordinateChannel1D, self).__init__(
            rank=1,
            use_radius=False,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel1D, self).get_config()
        config.pop('rank')
        config.pop('use_radius')
        return config


class CoordinateChannel2D(_CoordinateChannel):
    """ Adds Coordinate Channels to the input tensor.

    # Arguments
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(samples, channels + 2/3, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels + 2/3)`
        if `data_format` is `"channels_last"`.

        If `use_radius` is set, then will have 3 additional filers,
        else only 2 additional filters will be added.

    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, use_radius=False,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel2D, self).__init__(
            rank=2,
            use_radius=use_radius,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel2D, self).get_config()
        config.pop('rank')
        return config
