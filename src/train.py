# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
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

# train.py -- Train all or only part of the model from scratch or an
#   existing checkpoint.

import tensorflow as tf
import pipeline
import charset
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
tf.app.flags.DEFINE_integer('save_checkpoint_secs', 30,
                            """Interval between daving checkpoints""")

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
tf.app.flags.DEFINE_boolean('bucket_data',True,
                            """Bucket training data by width for efficiency""")


# For displaying various statistics while training
tf.logging.set_verbosity( tf.logging.INFO )

def _get_input():
    """
    Get tf.data.Dataset according to command-line flags for training 
    using tf.estimator.Estimator

    Note: Default behavior is bucketing according to default bucket boundaries
    listed in pipeline.get_data

    Returns:
      dataset : elements structured as [features, labels]
                feature structure can be seen in postbatch_fn 
                in mjsynth.py or maptextsynth.py for static or dynamic
                data pipelines respectively
    """

    # We only want a filter_fn if we have dynamic data (for now)
    filter_fn = None if FLAGS.static_data else filters.dyn_filter_by_width

    # Pack keyword arguments into dictionary
    data_args = { 'base_dir': FLAGS.train_path,
                  'file_patterns': str.split(FLAGS.filename_pattern, ','),
                  'num_threads': FLAGS.num_input_threads,
                  'batch_size': FLAGS.batch_size,
                  'input_device': FLAGS.input_device,
                  'filter_fn': filter_fn }

    if not FLAGS.bucket_data:
        data_args['boundaries']=None # Turn off bucketing (on by default)
        
    # Get data according to flags
    dataset = pipeline.get_data( FLAGS.static_data, **data_args)

    return dataset


def _get_config():
    """Setup config to soften device placement and set chkpt saving intervals"""

    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    custom_config = tf.estimator.RunConfig(session_config=device_config,
                                           save_checkpoints_secs=
                                           FLAGS.save_checkpoint_secs)

    return custom_config 


def main( argv=None ):

    # Set up a dictionary of arguments to be passed for training
    train_args = {'scope': FLAGS.tune_scope, 
                  'tune_from': FLAGS.tune_from, 
                  'train_device': FLAGS.train_device, 
                  'learning_rate': FLAGS.learning_rate, 
                  'decay_steps': FLAGS.decay_steps, 
                  'decay_rate': FLAGS.decay_rate, 
                  'decay_staircase': FLAGS.decay_staircase, 
                  'momentum':FLAGS.momentum}

    # Initialize the classifier
    classifier = tf.estimator.Estimator( config=_get_config(), 
                                         model_fn=model_fn.train_fn(
                                             **train_args),
                                         model_dir=FLAGS.output )
   
    # Train the model
    classifier.train( input_fn=_get_input, max_steps=FLAGS.max_num_steps )

if __name__ == '__main__':
    tf.app.run()
