import tensorflow as tf 
import model
import mjsynth
import flags
import charset

FLAGS = tf.app.flags.FLAGS
optimizer = 'Adam'

def _get_training( rnn_logits,label,sequence_length ):
    """Set up training ops"""
    with tf.name_scope( "train" ):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope )        

        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):

            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.momentum)
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate, 
                optimizer=optimizer,
                variables=rnn_vars)

            tf.summary.scalar( 'learning_rate', learning_rate )

    return train_op, loss


def _get_testing(rnn_logits,sequence_length,label,label_length):
    """Create ops for testing (all scalars): 
       loss: CTC loss function value, 
       label_error:  Batch-normalized edit distance on beam search max
       sequence_error: Batch-normalized sequence error rate
    """
    with tf.name_scope("train"):
        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False)
        sequence_errors = tf.count_nonzero(label_errors,axis=0)
        total_label_error = tf.reduce_sum( label_errors )
        total_labels = tf.reduce_sum( label_length )
        label_error = tf.truediv( total_label_error, 
                                  tf.cast(total_labels, tf.float32 ),
                                  name='label_error')
        sequence_error = tf.truediv( tf.cast( sequence_errors, tf.int32 ),
                                     tf.shape(label_length)[0],
                                     name='sequence_error')
        tf.summary.scalar( 'loss', loss )
        tf.summary.scalar( 'label_error', label_error )
        tf.summary.scalar( 'sequence_error', sequence_error )
    return loss, label_error, sequence_error


def model_fn (features, labels, mode):
    """Model function for the estimator object"""

    image = features['image']
    width = features['width']

    #NOT SURE WHAT DEVICE TO PUT THESE COMPUTATIONS IN
    conv_features,sequence_length = model.convnet_layers( image, 
                                                          width, 
                                                          mode)
    logits = model.rnn_layers(conv_features, sequence_length,
                                   charset.num_classes())

    #Training the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device(FLAGS.train_device):
            train_op, loss = _get_training(logits,labels,sequence_length)
            return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op)

    #Testing the model
    elif mode == tf.estimator.ModeKeys.EVAL:
        with tf.device(FLAGS.device):
            label = labels
            length = features['length']

            loss,label_error,sequence_error = _get_testing(
                logits,sequence_length,label,length)

            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              eval_metric_ops=
                                              {'label_error':
                                               label_err_metric_fn(label_error),
                                               'sequence_error':
                                               seq_err_metric_fn(sequence_error)},
                                              train_op=None)

def label_err_metric_fn(label_error):
    metric, update_op = tf.metrics.mean(label_error)

    return metric, update_op

def seq_err_metric_fn(sequence_error):
    metric, update_op = tf.metrics.mean(sequence_error)

    return metric, update_op

