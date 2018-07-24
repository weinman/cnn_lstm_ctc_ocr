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

# evaluate.py -- Streams evaluation statistics (i.e., character error
#   rate, sequence error rate) for a single batch whenever a new model
#   checkpoint appears

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import six
import model_fn
import pipeline

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer( 'batch_size',2**9,
                             """Eval batch size""" )
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                             """Time between test runs""")

tf.app.flags.DEFINE_string( 'device','/gpu:0',
                            """Device for graph placement""" )

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for event logs and checkpoints""" )
tf.app.flags.DEFINE_string( 'test_path','../data/',
                            """Base directory for test/validation data""" )
tf.app.flags.DEFINE_string( 'filename_pattern','val/words-*',
                            """File pattern for input data""" )
tf.app.flags.DEFINE_integer( 'num_input_threads',4,
                             """Number of readers for input data""" )
tf.app.flags.DEFINE_boolean( 'static_data', True,
                             """Whether to use static data 
                             (false for dynamic data)""" )

tf.logging.set_verbosity( tf.logging.WARN )
#tf.logging.set_verbosity( tf.logging.INFO )

def _get_input():
    """
    Get dataset according to tf flags for training using Estimator
    Note: Default behavior is bucketing according to default bucket boundaries
    listed in pipeline.get_data
    Returns:
      features, labels
                feature structure can be seen in postbatch_fn 
                in mjsynth.py or maptextsynth.py for static or dynamic
                data pipelines respectively
    """

    # We only want a filter_fn if we have dynamic data (for now)
    filter_fn = None if FLAGS.static_data else filters.dyn_filter_by_width

    # Get data according to flags
    dataset = pipeline.get_data( FLAGS.static_data,
                                 base_dir=FLAGS.test_path,
                                 file_patterns=str.split(
                                     FLAGS.filename_pattern,
                                     ','),
                                 num_threads=FLAGS.num_input_threads,
                                 batch_size=FLAGS.batch_size,
                                 input_device=FLAGS.device,
                                 filter_fn=filter_fn )

    iterator = dataset.make_one_shot_iterator()

    # Transforming the input into proper format
    features, labels = iterator.get_next()

    return features, labels


# Taken from the official source code of Tensorflow
# Licensed under the Apache License, Version 2.0
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/estimator.py
def _extract_metric_update_ops( eval_dict ):
  """Separate update operations from metric value operations."""
  update_ops = []
  value_ops = {}
  # Sort metrics lexicographically so graph is identical every time.
  for name, metric_ops in sorted( six.iteritems( eval_dict ) ):
    value_ops[name] = metric_ops[0]
    update_ops.append( metric_ops[1] )

  if update_ops:
    update_op = control_flow_ops.group( *update_ops )
  else:
    update_op = None

  return update_op, value_ops


def _get_config():
    """Setup session config to soften device placement"""
    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return device_config


def main(argv=None):
    
    # Get input tensors for evaluation
    features, labels = _get_input()

    # Construct the evaluation function 
    evaluate_fn = model_fn.evaluate_fn(FLAGS.device)

    # Wrap the ops in an Estimator spec object
    estimator_spec = evaluate_fn(features, labels, 
                                 tf.estimator.ModeKeys.EVAL, 
                                 {'continuous_eval': True})

    # Extract the necessary ops and the final tensors from the estimator spec
    update_op, value_ops = _extract_metric_update_ops(
        estimator_spec.eval_metric_ops)
  
    # Specify to evaluate N number of batches (in this case N==1)
    stop_hook = tf.contrib.training.StopAfterNEvalsHook( 1 )

    # Evaluate repeatedly once a new checkpoint is found
    tf.contrib.training.evaluate_repeatedly(
        checkpoint_dir=FLAGS.model,eval_ops=update_op, final_ops=value_ops, 
        hooks = [stop_hook], config=_get_config(), 
        eval_interval_secs= FLAGS.eval_interval_secs )
  
  
if __name__ == '__main__':
    tf.app.run()
