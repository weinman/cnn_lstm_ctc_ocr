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

import tensorflow as tf 
import model
import mjsynth
<<<<<<< HEAD
import flags
import charset
=======
import pipeline
>>>>>>> estimator_metrics_testing

def _get_image_info( features, mode ):
    """Calculates the logits and sequence length"""

    image = features['image']
    width = features['width']

    conv_features,sequence_length = model.convnet_layers( image, 
                                                          width, 
                                                          mode )

    logits = model.rnn_layers( conv_features, sequence_length,
                               pipeline.num_classes() )

    return logits, sequence_length

<<<<<<< HEAD
def _get_training( rnn_logits, label, sequence_length ):
    """Set up training ops"""
    with tf.name_scope( "train" ):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope )        

        loss = model.ctc_loss_layer( rnn_logits, label, sequence_length ) 

        # Update batch norm stats: http://stackoverflow.com/questions/43234667
        extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )

        with tf.control_dependencies( extra_update_ops ):
=======

def _get_init_pretrained( tune_from ):
    """Return lambda for reading pretrained initial model"""
    
    if not tune_from:
        return None
    
    # Extract the global variables
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ) )
    
    ckpt_path=tune_from

    # Function to build the scaffold to initialize the training process
    init_fn = lambda sess: saver_reader.restore( sess, ckpt_path )

    return init_fn


def _get_training( rnn_logits,label,sequence_length, tune_scope, 
                   learning_rate, decay_steps, decay_rate, decay_staircase, 
                   momentum ):
    """Set up training ops"""

    with tf.name_scope( "train" ):

        if tune_scope:
            scope=tune_scope
        else:            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope )        

        loss = model.ctc_loss_layer( rnn_logits,label,sequence_length ) 

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
>>>>>>> estimator_metrics_testing

        with tf.control_dependencies( extra_update_ops ):
            
            # Calculate the learning rate given the parameters
            learning_rate_final = tf.train.exponential_decay(
                learning_rate,
                tf.train.get_global_step(),
<<<<<<< HEAD
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate' )

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.momentum )
=======
                decay_steps,
                decay_rate,
                staircase=decay_staircase,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate_final,
                beta1=momentum )
>>>>>>> estimator_metrics_testing
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate_final, 
                optimizer=optimizer,
                variables=rnn_vars )

            tf.summary.scalar( 'learning_rate', learning_rate )

    return train_op, loss


<<<<<<< HEAD
def _get_testing( rnn_logits, sequence_length, label, label_length ):
    """
    Create ops for testing (all scalars): 
=======
def _get_testing( rnn_logits,sequence_length,label,label_length ):
    """Create ops for testing (all scalars): 
>>>>>>> estimator_metrics_testing
       loss: CTC loss function value, 
       label_error:   edit distance on beam search max
       sequence_error: sequence error rate
    """

    with tf.name_scope( "train" ):
        loss = model.ctc_loss_layer( rnn_logits,label,sequence_length ) 
    with tf.name_scope( "test" ):
        predictions,_ = tf.nn.ctc_beam_search_decoder( rnn_logits, 
                                                       sequence_length,
                                                       beam_width=128,
                                                       top_paths=1,
                                                       merge_repeated=True )

        hypothesis = tf.cast( predictions[0], tf.int32 ) # for edit_distance

        # Per-sequence statistic
        num_label_errors = tf.edit_distance( hypothesis, label, 
                                             normalize=False )

        # Per-batch summary counts
        batch_num_label_errors = tf.reduce_sum( num_label_errors)
        batch_num_sequence_errors = tf.count_nonzero( num_label_errors, axis=0 )
        batch_num_labels = tf.reduce_sum( label_length )
        
        # Wide integer type casts (prefer unsigned, but truediv dislikes those)
        batch_num_label_errors = tf.cast( batch_num_label_errors, tf.int64 )
        batch_num_sequence_errors = tf.cast( batch_num_sequence_errors, 
                                             tf.int64 )
        batch_num_labels = tf.cast( batch_num_labels, tf.int64)
        
    return loss, batch_num_label_errors, batch_num_sequence_errors, \
        batch_num_labels, predictions


def _get_label_err_ops( batch_num_label_error, batch_total_labels ):
    """Calculates the label error by accumulating for batches and returns
    the average"""

    var_collections=[tf.GraphKeys.LOCAL_VARIABLES]

    # Variables to tally across batches (all initially zero)
    total_num_label_errors = tf.Variable( 0, trainable=False,
                                          name='total_num_label_errors',
                                          dtype=tf.int64,
                                          collections=var_collections )

    total_num_labels =  tf.Variable( 0, trainable=False,
                                     name='total_num_labels',
                                     dtype=tf.int64,
                                     collections=var_collections )

    # Create the "+=" update ops and group together as one
    update_label_errors    = tf.assign_add( total_num_label_errors,
                                            batch_num_label_error )
    update_num_labels     = tf.assign_add( total_num_labels,
                                           batch_total_labels )

    update_op = tf.group(update_label_errors,update_num_labels )

    
    # Get the average label error across all inputs
    label_error = tf.truediv( total_num_label_errors, 
                              total_num_labels,
                              name='label_error' )    
   
    return label_error, update_op, total_num_label_errors, total_num_labels


def _get_seq_err_ops( batch_num_sequence_errors, label_length ):
    """Calculates the sequence error by accumulating for batches and returns
    the average"""

    var_collections=[tf.GraphKeys.LOCAL_VARIABLES]

    # Variables to tally across batches (all initially zero)
    total_num_sequence_errors =  tf.Variable( 0, trainable=False,
                                              name='total_num_sequence_errors',
                                              dtype=tf.int64,
                                              collections=var_collections )

    total_num_sequences =  tf.Variable( 0, trainable=False,
                                        name='total_num_sequences',
                                        dtype=tf.int64,
                                        collections=var_collections )

    # Get the batch size and cast it appropriately
    batch_size = tf.shape( label_length )[0]
    batch_size = tf.cast( batch_size, tf.int64 )

    # Create the "+=" update ops and group together as one
    update_sequence_errors = tf.assign_add( total_num_sequence_errors,
                                            batch_num_sequence_errors )
    update_num_sequences   = tf.assign_add( total_num_sequences,
                                            batch_size )

    update_op = tf.group(update_sequence_errors, update_num_sequences)

    # Get the average sequence error across all inputs
    sequence_error = tf.truediv( total_num_sequence_errors,
                                 total_num_sequences,
                                 name='sequence_error' )
    
    return sequence_error, update_op, total_num_sequence_errors,\
        total_num_sequences


def _get_dictionary_tensor( dictionary_path, charset ):
    return tf.sparse_tensor_to_dense( tf.to_int32(
	dictionary_from_file( dictionary_path, charset )))


def _get_output( rnn_logits,sequence_length, lexicon ):
    """Create ops for validation
       predictions: Results of CTC beam search decoding
    """
<<<<<<< HEAD
    with tf.name_scope( "train" ):
        loss = model.ctc_loss_layer( rnn_logits, label, sequence_length ) 
    with tf.name_scope( "test" ):
        predictions,_ = tf.nn.ctc_beam_search_decoder( rnn_logits, 
                                                       sequence_length,
                                                       beam_width=128,
                                                       top_paths=1,
                                                       merge_repeated=True )
        hypothesis = tf.cast( predictions[0], tf.int32 ) # for edit_distance
        label_errors = tf.edit_distance( hypothesis, label, normalize=False )
        sequence_errors = tf.count_nonzero( label_errors,axis=0 )
        total_label_error = tf.reduce_sum( label_errors )
        total_labels = tf.reduce_sum( label_length )
        label_error = tf.truediv( total_label_error, 
                                  tf.cast( total_labels, tf.float32 ),
                                  name='label_error')
        sequence_error = tf.truediv( tf.cast( sequence_errors, tf.int32 ),
                                     tf.shape( label_length )[0],
                                     name='sequence_error')
        tf.summary.scalar( 'loss', loss )
        tf.summary.scalar( 'label_error', label_error )
        tf.summary.scalar( 'sequence_error', sequence_error )
    return loss, label_error, sequence_error


def model_fn ( features, labels, mode ):
    """Model function for the estimator object"""

    image = features['image']
    width = features['width']

    #NOT SURE WHAT DEVICE TO PUT THESE COMPUTATIONS IN
    conv_features,sequence_length = model.convnet_layers( image, 
                                                          width, 
                                                          mode )
    
    logits = model.rnn_layers( conv_features, sequence_length,
                                   charset.num_classes() )

    # Training the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device( FLAGS.train_device ):
            train_op, loss = _get_training( logits, labels, sequence_length )
            return tf.estimator.EstimatorSpec( mode=mode, 
                                               loss=loss, 
                                               train_op=train_op )

    # Testing the model
    elif mode == tf.estimator.ModeKeys.EVAL:
        with tf.device( FLAGS.device ):
            label = labels
            length = features['length']

            loss,label_error,sequence_error = _get_testing( logits,
                                                            sequence_length,
                                                            label,
                                                            length )

            return tf.estimator.EstimatorSpec(
                mode=mode, 
                loss=loss, 
                eval_metric_ops={
                    'label_error':
                    label_err_metric_fn( label_error ),
                    'sequence_error':
                    seq_err_metric_fn( sequence_error )
                },
                train_op=None )

def label_err_metric_fn( label_error ):
    metric, update_op = tf.metrics.mean( label_error )
=======
    with tf.name_scope("test"):
	if lexicon:
	    dict_tensor = _get_dictionary_tensor( FLAGS.lexicon, 
                                                  pipeline.out_charset )
	    predictions,_ = tf.nn.ctc_beam_search_decoder_trie( rnn_logits,
                                                                sequence_length,
                                                                alphabet_size=
                                                                pipeline.
                                                                num_classes() ,
                                                                dictionary=
                                                                dict_tensor,
                                                                beam_width=128,
                                                                top_paths=1,
                                                                merge_repeated=
                                                                True )
	else:
	    predictions,_ = tf.nn.ctc_beam_search_decoder( rnn_logits,
                                                           sequence_length,
                                                           beam_width=128,
                                                           top_paths=1,
                                                           merge_repeated=True )
    return predictions


def train_fn( scope, tune_from, train_device, learning_rate, 
                    decay_steps, decay_rate, decay_staircase, momentum ): 
    """Returns a function that trains the model"""

    def train( features, labels, mode ):

        with tf.device( train_device ):
            logits, sequence_length = _get_image_info( features, mode )

            train_op, loss = _get_training( logits,labels,
                                            sequence_length, 
                                            scope, learning_rate, 
                                            decay_steps, decay_rate, 
                                            decay_staircase, momentum )

            scaffold = tf.train.Scaffold( init_fn=
                                          _get_init_pretrained( tune_from ) )

            return tf.estimator.EstimatorSpec( mode=mode, 
                                               loss=loss, 
                                               train_op=train_op,
                                               scaffold=scaffold )
    return train


def evaluate_fn( device ):
    """Returns a function that evaluates the model for all batches at once or 
    continuously for one batch"""

    def evaluate( features, labels, mode ):
                
        with tf.device( device ): 
            logits, sequence_length = _get_image_info( features, mode )

            continuous_eval = features['continuous_eval']
            label = features['label']
            length = features['length']

            # Get the predictions
            loss,\
                batch_label_error,\
                batch_sequence_error, \
                batch_total_labels, \
                _ = _get_testing( logits,sequence_length,label,
                                  length )

            # Getting the mean label errors
            mean_label_error, \
                update_op_label, \
                total_num_label_errors, \
                total_num_labels= _get_label_err_ops( batch_label_error, 
                                                      batch_total_labels )
            
            #Getting the mean sequence errors
            mean_sequence_error,\
                update_op_seq,\
                total_num_sequence_errs,\
                total_num_sequences= _get_seq_err_ops( batch_sequence_error, 
                                                       length )
            
            # Print the metrics while doing continuous evaluation (evaluate.py) 
            # Note: tf.Print is identical to tf.identity, except it prints
            # the list of metrics as a side effect
            if (continuous_eval):
                global_step = tf.train.get_or_create_global_step()
                mean_sequence_error = tf.Print( mean_sequence_error, 
                                                [global_step,
                                                loss,
                                                mean_label_error,
                                                mean_sequence_error] )

            # Create summaries for the approprite metrics
            tf.summary.scalar( 'loss', loss)
            tf.summary.scalar( 'mean_label_error', mean_label_error)
            tf.summary.scalar( 'mean_sequence_error', mean_sequence_error )

            # Convert to tensor in order to pass it to eval_metric_ops
            total_num_label_errors = tf.convert_to_tensor(
                total_num_label_errors)
            total_num_labels = tf.convert_to_tensor(
                total_num_labels)
            total_num_sequence_errs = tf.convert_to_tensor(
                total_num_sequence_errs)
            total_num_sequences = tf.convert_to_tensor(
                total_num_sequences)
            

            # All the eval_metric_ops that will be passed on to the 
            # EstimatorSpec object
            eval_metric_ops = {'mean_label_error': 
                               ( mean_label_error, 
                                 update_op_label ),
                               'mean_sequence_error': 
                               ( mean_sequence_error,
                                 update_op_seq ),
                               'total_num_label_errors': 
                               ( total_num_label_errors,
                                 tf.no_op() ),
                               'total_num_labels':
                               ( total_num_labels,
                                 tf.no_op() ),
                               'total_num_sequence_errs':
                               ( total_num_sequence_errs,
                                 tf.no_op() ),
                               'total_num_sequences':
                               ( total_num_sequences, 
                                 tf.no_op() )}
            
            return tf.estimator.EstimatorSpec( mode=mode, 
                                               loss=loss, 
                                               eval_metric_ops=eval_metric_ops )
    return evaluate


def predict_fn( device, lexicon ):
    """Returns a function that validates the input data"""

    def predict( features, labels, mode ):

         # Get the appropriate tensors
        image = features
        width = tf.size( image[1] )

        # Pre-process the images
        proc_image = mjsynth.preprocess_image( image )
        proc_image = tf.reshape( proc_image,[1,32,-1,1] ) # Make first dim batch

        #Pack the modified image data into a dictionary
        proc_img_data = {'image': proc_image, 'width': width}

        with tf.device( device ):
            logits, sequence_length = _get_image_info(proc_img_data, mode)
>>>>>>> estimator_metrics_testing

            prediction = _get_output( logits,sequence_length, lexicon )

<<<<<<< HEAD
def seq_err_metric_fn( sequence_error ):
    metric, update_op = tf.metrics.mean( sequence_error )
=======
            # predictions only takes dense tensors
            final_pred = tf.sparse_to_dense( prediction[0].indices, 
                                             prediction[0].dense_shape, 
                                             prediction[0].values, 
                                             default_value=0 ) 
>>>>>>> estimator_metrics_testing

        return tf.estimator.EstimatorSpec( mode=mode,predictions=( final_pred ))

    return predict
