# CNN-LSTM-CTC-OCR
# Copyright (C) 2019 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf



# The procedure dense_to_sparse_tight is a derivative work of the function
# dense_to_sparse from tensorflow.contrib.layers.python.layers.layers
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ \
# /layers/python/layers/layers.py which bears the following copyright notice:
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# The original copyrighted work is licensed under Apache License 2.0, which is
# available at http://www.apache.org/licenses/LICENSE-2.0, while the
# derivative work is licensed under GPLv3, to the maximum extent warranted by
# the original license.

def dense_to_sparse_tight(tensor, eos_token=0,
                          outputs_collections=None, scope=None):
    """Converts a dense tensor into a sparse tensor whose shape is no larger than 
     strictly necessary.
    Args:
     tensor: An `int` `Tensor` to be converted to a `Sparse`.
     eos_token: An integer.
       It is part of the target label that signifies the end of a sentence.
     outputs_collections: Collection to add the outputs.
     scope: Optional scope for name_scope.
    """
    with tf.compat.v1.variable_scope(scope, 'dense_to_sparse_tight', [tensor]) as sc:
        tensor = tf.convert_to_tensor(value=tensor)
        indices = tf.compat.v1.where(
            tf.math.not_equal(tensor, tf.constant(eos_token,
                                                tensor.dtype)))
        # Need to verify there are *any* indices that are not eos_token
        # If none, give shape [1,0].
        shape = tf.cond( pred=tf.not_equal(tf.shape(input=indices)[0],
                                      tf.constant(0)), # Found valid indices?
                         true_fn=lambda: tf.cast(tf.reduce_max(input_tensor=indices,axis=0),\
                                                 tf.int64) + 1,
                         false_fn=lambda: tf.cast([1,0], tf.int64) )
        values = tf.gather_nd(tensor, indices)
        outputs = tf.SparseTensor(indices, values, shape)
        return layers_utils.collect_named_outputs(outputs_collections,
                                                  sc.name, outputs)
    
