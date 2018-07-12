import tensorflow as tf
from tensorflow.contrib.training.python.training import evaluation
import flags
import model_fn
import pipeline
from tensorflow.python.estimator import util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.estimator import model_fn as model_fn_lib
import six
from tensorflow.python.ops import control_flow_ops

FLAGS = tf.app.flags.FLAGS

optimizer = 'Adam'
tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)


def _extract_metric_update_ops(eval_dict):
  """Separate update operations from metric value operations."""
  update_ops = []
  value_ops = {}
  # Sort metrics lexicographically so graph is identical every time.
  for name, metric_ops in sorted(six.iteritems(eval_dict)):
    value_ops[name] = metric_ops[0]
    update_ops.append(metric_ops[1])

  if update_ops:
    update_op = control_flow_ops.group(*update_ops)
  else:
    update_op = None

  return update_op, value_ops

def _get_input_stream():
    if(FLAGS.static_data):
        ds = pipeline.get_static_data(FLAGS.test_path, 
                                      str.split(
                                          FLAGS.filename_pattern_test,','),
                                      num_threads=FLAGS.num_input_threads_eval,
                                      boundaries=None, # No bucketing
                                      batch_size=FLAGS.batch_size_eval,
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

def _get_session_config():
    """Setup session config to soften device placement"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False,
        gpu_options=gpu_options)

    return config


def main(argv=None):
    

    features, labels = _get_input_stream()

    estimator_spec = model_fn.model_fn(features, labels, 
                                       tf.estimator.ModeKeys.EVAL)

    update_op, eval_dict = _extract_metric_update_ops(
        estimator_spec.eval_metric_ops)
    
    hooks = tf.contrib.training.StopAfterNEvalsHook(1)

    tf.contrib.training.evaluate_repeatedly(
        checkpoint_dir=FLAGS.model,eval_ops=update_op, final_ops=eval_dict, 
        hooks = [hooks], config=_get_session_config())

if __name__ == '__main__':
    tf.app.run()
