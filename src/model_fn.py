import tensorflow as tf 
import model
import mjsynth
import flags
import pipeline

FLAGS = tf.app.flags.FLAGS
optimizer = 'Adam'

tf.logging.set_verbosity(tf.logging.INFO)

def _get_training(rnn_logits,label,sequence_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)        

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

        # Per-sequence statistic
        num_label_errors = tf.edit_distance(hypothesis, label, normalize=False)
        # Per-batch summary counts
        batch_num_label_errors = tf.reduce_sum( num_label_errors, name='damn')
        batch_num_sequence_errors = tf.count_nonzero(num_label_errors,axis=0)
        batch_num_labels = tf.reduce_sum( label_length )
        
        # Wide integer type casts (prefer unsigned, but truediv dislikes those)
        batch_num_label_errors = tf.cast( batch_num_label_errors, tf.int64 )
        batch_num_sequence_errors = tf.cast(batch_num_sequence_errors, tf.int64)
        batch_num_labels = tf.cast( batch_num_labels, tf.int64)
        tf.identity(batch_num_label_errors, name='lab')
        
    return loss, batch_num_label_errors, batch_num_sequence_errors, \
        batch_num_labels, predictions


def model_fn (features, labels, mode):
    """Model function for the estimator object"""

    image = features['image']
    width = features['width']

    #NOT SURE WHAT DEVICE TO PUT THESE COMPUTATIONS IN
    conv_features,sequence_length = model.convnet_layers( image, 
                                                          width, 
                                                          mode)
    logits = model.rnn_layers(conv_features, sequence_length,
                                   pipeline.num_classes())

    #Training the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device(FLAGS.train_device):
            train_op, loss = _get_training(logits,labels,sequence_length)
            return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op)

    #Evaluating the model
    elif mode == tf.estimator.ModeKeys.EVAL:

        with tf.device(FLAGS.device):  
            continuous_eval = features['continuous_eval']
            label = features['label']
            length = features['length']

            if (continuous_eval == False):
                filename = features['filename']
                text = features['text']

            loss,\
                label_error,\
                sequence_error, \
                total_labels, \
                predictions = _get_testing(logits,sequence_length,label,length)

            # Getting the label errors
            mean_label_error, \
                update_op_label, \
                total_num_label_errors, \
                total_num_labels= label_err_metric_fn(label_error, total_labels)
       
            #Getting the sequence errors
            mean_sequence_error,\
                update_op_seq,\
                total_num_sequence_errors,\
                total_num_sequences= seq_err_metric_fn(sequence_error, length)
            
            if (continuous_eval == True):
                global_step = tf.train.get_or_create_global_step()
                mean_sequence_error = tf.Print(mean_sequence_error, 
                                               [loss,
                                                mean_sequence_error,
                                                mean_label_error,
                                                global_step])
            
            # Stack it into one tensor
            metrics = tf.convert_to_tensor(tf.stack([total_num_label_errors,
                                                     total_num_labels,
                                                     total_num_sequence_errors,
                                                     total_num_sequences], 
                                                    axis=0))

            if FLAGS.verbose == True:
                #Get the list of extra information 
                file_list, update_op_fname = filename_metric_fn(filename, \
                                                                text, \
                                                                predictions)

                return tf.estimator.EstimatorSpec(mode=mode, 
                                                  loss=loss, 
                                                  eval_metric_ops=
                                                  {'label_error':
                                                   (mean_label_error, 
                                                    update_op_label),
                                                   'sequence_error':
                                                   (mean_sequence_error,
                                                    update_op_seq),
                                                   'misc_metrics':\
                                                   (metrics, metrics),
                                                   'filename':\
                                                   (file_list, update_op_fname)})
            else:
                return tf.estimator.EstimatorSpec(mode=mode, 
                                                  loss=loss, 
                                                  eval_metric_ops=
                                                  {'label_error':
                                                   (mean_label_error, 
                                                    update_op_label),
                                                   'sequence_error':
                                                   (mean_sequence_error,
                                                    update_op_seq),
                                                   'misc_metrics':\
                                                   (metrics, metrics)})


def filename_metric_fn(filename, text, predictions):
    var_collections=[tf.GraphKeys.LOCAL_VARIABLES]

    # Variable that holds the list for info if FLAGS.verbose is True
    file_list = tf.Variable([], 
                            expected_shape=tf.TensorShape([None]), 
                            dtype=tf.string,collections=var_collections)

    # Update the file list
    single_file = tf.concat([text, filename], axis=0)
    update_op = tf.concat([file_list, single_file], axis = 0)

    # Final list
    lst_tensor = tf.convert_to_tensor(file_list)

    return lst_tensor, update_op

def label_err_metric_fn(batch_num_label_error, batch_total_labels):
    
    var_collections=[tf.GraphKeys.LOCAL_VARIABLES]

    # Variables to tally across batches (all initially zero)
    total_num_label_errors = tf.Variable(0, trainable=False,
                                             name='total_num_label_errors',
                                             dtype=tf.int64,
                                         collections=var_collections)

    total_num_labels =  tf.Variable(0, trainable=False,
                                        name='total_num_labels',
                                        dtype=tf.int64,
                                    collections=var_collections)

    # Create the "+=" update ops and group together as one
    update_label_errors    = tf.assign_add( total_num_label_errors,
                                            batch_num_label_error)
    update_num_labels     = tf.assign_add( total_num_labels,
                                            batch_total_labels )

    update_op = tf.group(update_label_errors,update_num_labels )

    
    # Get the average label error across all inputs
    label_error = tf.truediv( total_num_label_errors, 
                                  total_num_labels,
                              name='label_error')    
   
    return label_error, update_op, total_num_label_errors, total_num_labels

def seq_err_metric_fn(batch_num_sequence_errors, label_length):

    var_collections=[tf.GraphKeys.LOCAL_VARIABLES]

    # Variables to tally across batches (all initially zero)
    total_num_sequence_errors =  tf.Variable(0, trainable=False,
                                    name='total_num_sequence_errors',
                                    dtype=tf.int64,
                                    collections=var_collections)

    total_num_sequences =  tf.Variable(0, trainable=False,
                                       name='total_num_sequences',
                                       dtype=tf.int64,
                                       collections=var_collections)

    batch_size = tf.shape(label_length)[0]
    batch_size = tf.cast(batch_size, tf.int64)

    # Create the "+=" update ops and group together as one
    update_sequence_errors = tf.assign_add( total_num_sequence_errors,
                                            batch_num_sequence_errors )
    update_num_sequences   = tf.assign_add( total_num_sequences,
                                            batch_size)

    update_op = tf.group(update_sequence_errors, update_num_sequences)

    # Get the average sequence error across all inputs
    sequence_error = tf.truediv( total_num_sequence_errors,
                                 total_num_sequences,
                                 name='sequence_error')
    
    return sequence_error, update_op, total_num_sequence_errors,\
        total_num_sequences

