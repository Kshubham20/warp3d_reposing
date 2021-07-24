import tensorflow as tf
import math
import numpy as np
from skimage.measure import compare_ssim


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def format_time(s):
    if s < 60:
        return str(round(s, 4))
    else:
        m = int(s / 60)
        s = s - m * 60
        s = int(s)
        if m < 60:
            return str(m) + ':' + str(s)
        else:
            h = int(m / 60)
            m = m - h * 60
            return str(h) + ':' + str(m) + ':' + str(s)


def extract_paths_from_deep_dict(d, prefix=[]):
    if type(d) is not dict:
        return [prefix]
    else:
        return sum([extract_paths_from_deep_dict(d[key], prefix + [key]) for key in d], [])


def get_from_deep_dict(d, keys):
    for k in keys:
        d = d[k]
    return d


def ssim(tar, gen, masks=None):
    tar = np.array(tar)
    gen = np.array(gen)
    if masks is not None:
        masks = np.array(masks)
    data_range = max(gen.max(), tar.max()) - min(gen.min(), tar.min())
    ssims = []
    fg_ssims = []
    bg_ssims = []
    for t, g, m in zip(tar, gen, masks):
        fgc = np.sum(m)
        bgc = m.shape[0] * m.shape[1] - fgc
        s, si = compare_ssim(t, g, multichannel=True, data_range=data_range, full=True)
        ssims.append(s)
        fg_ssims.append(np.sum(si * m) / fgc / si.shape[-1])
        bg_ssims.append(np.sum(si * (1 - m)) / bgc / si.shape[-1])
    if masks is not None:
        return np.mean(ssims), np.mean(fg_ssims), np.mean(bg_ssims)
    else:
        return np.mean(ssims)


def soft_argmax(inp, axis):
    softmaxed = softmax(inp, axis=axis)
    return tf.stack(decode_heatmap(softmaxed, axis=axis), axis=-1)


def softmax(target, axis=-1, name=None):
    with tf.name_scope(name, 'softmax', values=[target]):
        max_along_axis = tf.reduce_max(target, axis, keepdims=True)
        exponentiated = tf.exp(target - max_along_axis)
        normalizer_denominator = tf.reduce_sum(exponentiated, axis, keepdims=True)
        return exponentiated / normalizer_denominator


def decode_heatmap(inp, axis=-1):
    shape = inp.get_shape().as_list()
    ndims = inp.get_shape().ndims

    def relative_coords_along_axis(ax):
        grid_shape = [1] * ndims
        grid_shape[ax] = shape[ax]
        grid = tf.reshape(tf.linspace(0.0, 1.0, shape[ax]), grid_shape)
        return tf.cast(grid, inp.dtype)

    # Single axis:
    if not isinstance(axis, (tuple, list)):
        return tf.reduce_sum(relative_coords_along_axis(axis) * inp, axis=axis)

    # Multiple axes.
    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else ndims + ax + 1 for ax in axis]
    result = []
    for ax in heatmap_axes:
        other_heatmap_axes = tuple(set(heatmap_axes) - {ax})
        summed_over_other_axes = tf.reduce_sum(inp, axis=other_heatmap_axes, keepdims=True)
        coords = relative_coords_along_axis(ax)
        decoded = tf.reduce_sum(coords * summed_over_other_axes, axis=ax, keepdims=True)
        result.append(tf.squeeze(decoded, heatmap_axes))

    return result


def make_pretrained_weight_loader(pretrained_path, loaded_scope, checkpoint_scope, excluded_parts, replace_names):
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=loaded_scope)
    var_dict = {v.op.name[v.op.name.index(checkpoint_scope):]: v for v in var_list}
    var_dict = {k: v for k, v in var_dict.items() if not any(excl in k for excl in excluded_parts)}
    for fr, to in replace_names:
        var_dict = {k.replace(fr, to): v for k, v in var_dict.items()}
    saver = tf.train.Saver(var_list=var_dict)

    # global_init_op = tf.global_variables_initializer()

    def init_fn(sess):
        # sess.run(global_init_op)
        saver.restore(sess, pretrained_path)

    return init_fn


def extend_spatial_sizes(t):
    return tf.pad(t, [[0, 0]] + [[0, 1]] * (len(t.shape) - 2) + [[0, 0]])


def reduce_spatial_sizes(t):
    for i in range(1, len(t.shape) - 1):
        t = tf.gather(t, list(range(1, int(t.shape[i]))), axis=i)
    return t

#background impainter
import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras.utils import conv_utils
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model


class PConv2D(Conv2D):
    def build(self, input_shape):        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        self.input_dim = input_shape[0][channel_axis]
        
        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(shape=kernel_shape)

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = ( (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),)

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None
        
    def call(self, inputs, mask=None):
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks  = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        mask_output = K.conv2d( masks, self.kernel_mask,  strides=self.strides, padding='valid', data_format=self.data_format, dilation_rate=self.dilation_rate)
        img_output  = K.conv2d((images*masks), self.kernel, strides=self.strides, padding='valid', data_format=self.data_format, dilation_rate=self.dilation_rate)        

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize image output
        img_output = img_output * mask_ratio

        if self.use_bias:
            img_output = K.bias_add( img_output, self.bias, data_format=self.data_format)
        
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]
    
img_rows=512
img_cols=512
inputs_img  = Input((img_rows, img_cols, 3))
inputs_mask = Input((img_rows, img_cols, 3))

def encoder(img_in, mask_in, filters, kernel_size, bn=True):
    conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
    if bn:
        conv = tfa.GroupNormalization(conv, training=True)
    conv = Activation('relu')(conv)
    return conv, mask
    
    e_conv1, e_mask1 = encoder(inputs_img, inputs_mask, 64, 7, bn=False)
    e_conv2, e_mask2 = encoder(e_conv1, e_mask1, 128, 5)
    e_conv3, e_mask3 = encoder(e_conv2, e_mask2, 256, 5)
    e_conv4, e_mask4 = encoder(e_conv3, e_mask3, 512, 3)
    e_conv5, e_mask5 = encoder(e_conv4, e_mask4, 512, 3)
    e_conv6, e_mask6 = encoder(e_conv5, e_mask5, 512, 3)
    e_conv7, e_mask7 = encoder(e_conv6, e_mask6, 512, 3)
    e_conv8, e_mask8 = encoder(e_conv7, e_mask7, 512, 3)
    
def decoder(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
    up_img      = UpSampling2D(size=(2,2))(img_in)
    up_mask     = UpSampling2D(size=(2,2))(mask_in)
    concat_img  = Concatenate(axis=3)([e_conv,up_img])
    concat_mask = Concatenate(axis=3)([e_mask,up_mask])

    conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])

    if bn:
        conv = tfa.GroupNormalization()(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    return conv, mask
        
    d_conv1, d_mask1 = decoder(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3)
    d_conv2, d_mask2 = decoder(d_conv1, d_mask1, e_conv6, e_mask6, 512, 3)
    d_conv3, d_mask3 = decoder(d_conv2, d_mask2, e_conv5, e_mask5, 512, 3)
    d_conv4, d_mask5 = decoder(d_conv3, d_mask3, e_conv4, e_mask4, 512, 3)
    d_conv5, d_mask5 = decoder(d_conv4, d_mask4, e_conv3, e_mask3, 256, 3)
    d_conv6, d_mask6 = decoder(d_conv5, d_mask5, e_conv2, e_mask2, 128, 3)
    d_conv7, d_mask7 = decoder(d_conv6, d_mask6, e_conv1, e_mask1, 64, 3)
    d_conv8, d_mask8 = decoder(d_conv7, d_mask7, inputs_img, inputs_mask, 3, 3, bn=False)
    outputs = Conv2D(3, 1, activation = 'sigmoid')(d_conv8)
    
    model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

    return model, inputs_mask    

