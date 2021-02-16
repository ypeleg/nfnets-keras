

import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D
from keras.activations import sigmoid


class WSConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(kernel_initializer = "he_normal", *args, **kwargs)

    def standardize_weight(self, weight, eps):
        mean = tf.math.reduce_mean(weight, axis = (0, 1, 2), keepdims = True)
        var = tf.math.reduce_variance(weight, axis = (0, 1, 2), keepdims = True)
        fan_in = np.prod(weight.shape[:-1])
        gain = self.add_weight(name = 'gain', shape = (weight.shape[-1],), initializer = lambda : tf.keras.initializers.Ones, trainable = True, dtype = self.dtype)
        scale = tf.math.rsqrt(tf.math.maximum(var * fan_in, tf.convert_to_tensor(eps, dtype = self.dtype))) * gain
        return weight * scale - (mean * scale)

    def call(self, inputs, eps = 1e-4):
        self.kernel.assign(self.standardize_weight(self.kernel, eps))
        return super().call(inputs)


class SqueezeExcite(tf.keras.Model):

    def __init__(self, in_ch, out_ch, se_ratio = 0.5, hidden_ch = None, activation = tf.keras.activations.relu, name = None):
        super(SqueezeExcite, self).__init__(name = name)
        self.in_ch, self.out_ch = in_ch, out_ch
        if se_ratio is None:
            if hidden_ch is None: raise ValueError('Must provide one of se_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else: self.hidden_ch = max(1, int(self.in_ch * se_ratio))
        self.activation = activation
        self.fc0 = tf.keras.layers.Dense(self.hidden_ch, use_bias = True)
        self.fc1 = tf.keras.layers.Dense(self.out_ch, use_bias = True)

    def call(self, x):
        h = tf.math.reduce_mean(x, axis = [1, 2])
        h = self.fc1(self.activation(self.fc0(h)))
        h = sigmoid(h)[:, None, None]
        return h


class StochasticDepth(Model):

    def __init__(self, drop_rate, scale_by_keep = False, name = None):
        super(StochasticDepth, self).__init__(name = name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def call(self, x, training):
        if not training: return x
        batch_size = x.shape[0]
        r = tf.random.uniform(shape = [batch_size, 1, 1, 1], dtype = x.dtype)
        keep_prob = 1. - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep: x = x / keep_prob
        return x * binary_tensor
