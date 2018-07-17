# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
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

import os
import tensorflow as tf
from tensorflow.contrib import learn

import pipeline
import model
import model_fn
import filters

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output','../data/model',
                          """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from','',
                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope','',
                          """Variable scope for training""")

tf.app.flags.DEFINE_integer('batch_size',2**5,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate',1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps',2**16,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_boolean('decay_staircase',False,
                          """Staircase learning rate decay by integer division""")


tf.app.flags.DEFINE_integer('max_num_steps', 2**21,
                            """Number of optimization steps to run""")
tf.app.flags.DEFINE_boolean('static_data', True,
                            """Whether to use static data 
                            (false for dynamic data)""")

tf.app.flags.DEFINE_string('train_device','/gpu:1',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device','/gpu:0',
                           """Device for preprocess/batching graph placement""")

tf.app.flags.DEFINE_string('train_path','../data/train/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern','words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4,
                          """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
"""Limit of input string length width""")

# For displaying various statistics while training
tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer='Adam'

def _get_input_stream():
    if(FLAGS.static_data):
        ds = pipeline.get_static_data(FLAGS.train_path, 
                                      str.split(
                                          FLAGS.filename_pattern,','),
                                      num_threads=FLAGS.num_input_threads,
                                      batch_size=FLAGS.batch_size,
                                      input_device=FLAGS.input_device,
                                      filter_fn=None)
                                    
    else:
        ds = pipeline.get_dynamic_data(num_threads=FLAGS.num_input_threads,
                                       batch_size=FLAGS.batch_size,
                                       input_device=FLAGS.input_device,
                                       filter_fn=filters.dyn_filter_by_width)

    iterator = ds.make_one_shot_iterator()

    if (FLAGS.static_data):
        image, width, label, _, _, _ = iterator.get_next()
    else:
        image, width, label, _, _ = iterator.get_next()

    # The input for the model function 
    features = {"image": image, "width": width, "optimizer": optimizer}
    
    return features, label

def _get_session_config():
    """Setup session config to soften device placement"""

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False,
        gpu_options=gpu_options)

    return config 

'''def model_fn_wrapper(FLAGS.tune_scope, FLAGS.train_device, FLAGS.learning_rate, FLAGS.decay_steps,
                     FLAGS.decay_rate, FLAGS.decay_staircase, FLAGS.momentum, FLAGS.output, 
                     FLAGS.max_num_steps):

    train_fn = lambda : modelfn.model_fn
    return train_fn'''

def main(argv=None):

    custom_config = tf.estimator.RunConfig(session_config=_get_session_config(),
                                           save_checkpoints_secs=30)
    flags = [FLAGS.output]
    # Initialize the classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn._train_wrapper(
        FLAGS.tune_scope, FLAGS.tune_from, FLAGS.train_device, 
        FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate, 
        FLAGS.decay_staircase, FLAGS.momentum),
                                        model_dir=FLAGS.output,
                                        config=custom_config)

    # Train the model
    classifier.train(input_fn=lambda: _get_input_stream(), 
                     max_steps=FLAGS.max_num_steps)

if __name__ == '__main__':
    tf.app.run()
