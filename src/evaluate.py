import tensorflow as tf
from tensorflow.contrib.training.python.training import evaluation
import model_fn
import pipeline
from tensorflow.python.estimator import util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.estimator import model_fn as model_fn_lib
import six
from tensorflow.python.ops import control_flow_ops

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer( 'batch_size',2**9,
                             """Eval batch size""" )
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                             'Time between test runs')

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
tf.logging.set_verbosity( tf.logging.INFO )

def _get_input_stream():
    if(FLAGS.static_data):
        ds = pipeline.get_static_data(FLAGS.test_path, 
                                      str.split(
                                        FLAGS.filename_pattern,','),
                                      num_threads=FLAGS.num_input_threads,
                                      boundaries=None, # No bucketing
                                      batch_size=FLAGS.batch_size,
                                      input_device=FLAGS.device,
                                      filter_fn=None)
                                    
    else:
        ds = pipeline.get_dynamic_data(num_threads=FLAGS.num_input_threads_eval,
                                       batch_size=FLAGS.batch_size_eval,
                                       boundaries=None, # No bucketing
                                       input_device=FLAGS.device,
                                       filter_fn=filters.dyn_filter_by_width)

    iterator = ds.make_one_shot_iterator() 
    
    image, width, label, length, _, _ = iterator.get_next()

    # The input for the model function 
    features = {"image": image, 
                "width": width, 
                "length": length, 
                "label": label,
                "continuous_eval": True}

    return features, label

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

def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config


def main(argv=None):
    
  #Input for evaluate function
  features, labels = _get_input_stream()

  # Returns a evaluation function 
  evaluate_fn = model_fn._evaluate_wrapper(FLAGS.device)

  # Wraps all the necessary ops in an Estimator spec object
  estimator_spec = evaluate_fn(features, labels, 
                               tf.estimator.ModeKeys.EVAL)

  # Extracts the necessary ops and the final tensors from the estimator spec
  update_op, eval_dict = _extract_metric_update_ops(
    estimator_spec.eval_metric_ops)
  
  # Hook responsible for evaluating X number of batches (in this case it is 1)
  hooks = tf.contrib.training.StopAfterNEvalsHook( 1 )

  # Evaluates repeatedly once a new checkpoint is found
  tf.contrib.training.evaluate_repeatedly(
    checkpoint_dir=FLAGS.model,eval_ops=update_op, final_ops=eval_dict, 
    hooks = [hooks], config=_get_session_config( ), 
    eval_interval_secs= eval_interval_secs )
  
  
if __name__ == '__main__':
    tf.app.run()
