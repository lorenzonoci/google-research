# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Contains models used in the experiments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf

from cold_posterior_bnn.core import frn
from cold_posterior_bnn.imdb import imdb_model

import numpy as np
from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool

def build_cnnlstm(num_words, sequence_length, pfac):
  model = imdb_model.cnn_lstm_nd(pfac, num_words, sequence_length)
  return model


def build_resnet_v1(input_shape, depth, num_classes, pfac, use_frn=False,
                    use_internal_bias=True, use_gconv=False):
  """Builds ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    num_classes: Number of output classes.
    pfac: priorfactory.PriorFactory class.
    use_frn: if True, then use Filter Response Normalization (FRN) instead of
      batchnorm.
    use_internal_bias: if True, use biases in all Conv layers.
      If False, only use a bias in the final Dense layer.

  Returns:
    tf.keras.Model.
  """
  def resnet_layer(inputs,
                   filters,
                   kernel_size=3,
                   strides=1,
                   activation=None,
                   pfac=None,
                   use_frn=False,
                   use_bias=True,
                   is_first_layer=False):
    """2D Convolution-Batch Normalization-Activation stack builder.

    Args:
      inputs: tf.Tensor.
      filters: Number of filters for Conv2D.
      kernel_size: Kernel dimensions for Conv2D.
      strides: Stride dimensinons for Conv2D.
      activation: tf.keras.activations.Activation.
      pfac: prior.PriorFactory object.
      use_frn: if True, use Filter Response Normalization (FRN) layer
      use_bias: if True, use biases in Conv layers.

    Returns:
      tf.Tensor.
    """
    x = inputs
    logging.info('Applying conv layer.')

    x = pfac(conv_layer(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=use_bias,
        use_gconv=use_gconv,
        is_first_layer=is_first_layer))(x)

    # elif use_gconv:
    #   x = GBatchNorm(h='D4')(x)
    if use_frn:
      x = pfac(frn.FRN())(x)
    else:
      x = tf.keras.layers.BatchNormalization()(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
    return x

  # Main network code
  num_res_blocks = (depth - 2) // 6
  filters = 16
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

  logging.info('Starting ResNet build.')
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = resnet_layer(inputs,
                   filters=filters,
                   activation='relu',
                   pfac=pfac,
                   use_frn=use_frn,
                   use_bias=use_internal_bias,
                   is_first_layer=True)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(x,
                       filters=filters,
                       strides=strides,
                       activation='relu',
                       pfac=pfac,
                       use_frn=use_frn,
                       use_bias=use_internal_bias)
      y = resnet_layer(y,
                       filters=filters,
                       activation=None,
                       pfac=pfac,
                       use_frn=use_frn,
                       use_bias=use_internal_bias)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = resnet_layer(x,
                         filters=filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         pfac=pfac,
                         use_frn=use_frn,
                         use_bias=use_internal_bias)
      x = tf.keras.layers.add([x, y])
      if use_frn:
        x = pfac(frn.TLU())(x)
      else:
        x = tf.keras.layers.Activation('relu')(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  if use_gconv:
      x = GroupPool(h_input='D4')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = pfac(tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal'))(x)

  logging.info('ResNet successfully built.')
  return tf.keras.models.Model(inputs=inputs, outputs=x)


def build_cnn(input_shape, depth, num_classes, pfac,
                    use_internal_bias=True, initializer='he_normal', activation='relu',
                    n_filters=32, kernel_size=3, use_gconv=False, g_group='D4',
                    strides=2, final_dense=False, double_n_filters=True, padding="same",
                    average_pooling=False, use_frn=False, use_batch_norm=True):
    
    if average_pooling and strides > 1:
        conv_strides = 1
    else:
        conv_strides = strides
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    logging.info('Starting CNN net')
    layer_strides = 1 # No subsampling at first layer
    for l in range(depth):
        is_first_layer = True if l==0 else False
        logging.info("Building Conv Layer")
        x = pfac(conv_layer(filters=n_filters,
                            kernel_size=kernel_size,
                            strides=layer_strides,
                            kernel_initializer=initializer,
                            use_bias=use_internal_bias,
                            use_gconv=use_gconv,
                            g_tranform=g_group,
                            padding=padding,
                            is_first_layer=is_first_layer))(x)
        #if use_gconv and use_batch_norm:
        #    x = GBatchNorm(h=g_group)(x)
        if use_frn:
            x = frn.FRN()(x)
        elif use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        if use_frn:
            x = frn.TLU()(x)
        else:
            x = tf.keras.layers.Activation(activation)(x)
        if l % 2 == 0 and average_pooling and not is_first_layer:
            x = tf.keras.layers.AveragePooling2D(pool_size=strides)(x)
        if l % 2 != 0: # downsample at next layer
            layer_strides = conv_strides
        else:
            layer_strides = 1
        if l % 2 != 0 and double_n_filters: # double filters at next layer
            n_filters *= 2
    if use_gconv:
        x = GroupPool(h_input=g_group)(x)
    logging.info("Group pooling added")
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    if final_dense:
        x = pfac(tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=initializer))(x)

    logging.info('ResNet successfully built.')
    return tf.keras.models.Model(inputs=inputs, outputs=x)


class TemperedLikelihoodWrapper(tf.keras.Model):

  def __init__(self, model, temp=1.0):
      super(TemperedLikelihoodWrapper, self).__init__()
      self.model = model
      self.likelihood_temp = tf.Variable(temp, trainable=False)

  def compile(self, optimizer, loss, **kwargs):
      super(TemperedLikelihoodWrapper, self).compile(**kwargs)
      self.optimizer = optimizer
      self.loss = loss

  def get_weights(self):
      return self.model.get_weights()

  def call(self, inputs, training=None, mask=None):
      return self.model(inputs, training=training)

  def train_step(self, data):

      x_batch_train, y_batch_train = data
      with tf.GradientTape(persistent=True) as tape:
          y_batch_train = tf.expand_dims(y_batch_train, axis=-1)
          logits = self(x_batch_train, training=True)  # model unnormalized probs
          log_prior_term = sum(self.model.losses)  # minus log prior
          ce = self.loss(y_batch_train, logits)
          obj = 1 / self.likelihood_temp * ce + log_prior_term
      grads = tape.gradient(obj, self.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


      self.compiled_metrics.update_state(y_batch_train, logits)
      return {m.name: m.result() for m in self.metrics}

  def get_config(self):
      return self.model.get_config()

  @property
  def trainable_variables(self):
    return self.model.trainable_variables


def conv_layer(
        filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer="he_normal",
        use_bias=True,
        g_tranform='D4',
        is_first_layer=False,
        depth_multiplier=None,
        use_gconv=False):
  if use_gconv:
      h_input = 'Z2' if is_first_layer else g_tranform
      if depth_multiplier is None:
          if g_tranform == 'D4':
              depth_multiplier = 1 / np.sqrt(8)
          elif g_tranform == 'C4':
              depth_multiplier = 1 / 2
          else:
              raise ValueError
      conv_layer_instance = GConv2D(
          int(round(depth_multiplier * filters)),
          kernel_size=kernel_size,
          h_input=h_input,
          h_output=g_tranform,
          strides=strides,
          padding=padding,
          kernel_initializer=kernel_initializer,
          use_bias=False
      )
  else:
      conv_layer_instance = tf.keras.layers.Conv2D(
                                        filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        kernel_initializer=kernel_initializer,
                                        use_bias=use_bias
      )
  return conv_layer_instance