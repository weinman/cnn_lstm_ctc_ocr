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

# model_fn.py -- Provides functions necessary for using the Estimator
#   API to control training, evaluation, and prediction.

import tensorflow as tf 
import model
import mjsynth
import charset
import pipeline
import utils

# Beam search width for prediction and evaluation modes using both the
# custom, lexicon-driven CTCWordBeamSearch module and the open-lexicon
# tf.nn.ctc_beam_search_decoder
_ctc_beam_width = 2**7

def _get_image_info( features, mode ):
    """Calculates the logits and sequence length"""

    image = features['image']
    width = features['width']

    conv_features,sequence_length = model.convnet_layers( image, 
                                                          width, 
                                                          mode )

    logits = model.rnn_layers( conv_features, sequence_length,
                               charset.num_classes() )

    return logits, sequence_length


def _get_init_pretrained( tune_from ):
    """Return lambda for reading pretrained initial model with a given session"""
    
    if not tune_from:
        return None
    
    # Extract the global variables
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ) )
    
    ckpt_path=tune_from

    # Function to build the scaffold to initialize the training process
    init_fn = lambda scaffold, sess: saver_reader.restore( sess, ckpt_path )

    return init_fn


def _get_training( rnn_logits,label,sequence_length, tune_scope, 
                   learning_rate, decay_steps, decay_rate, decay_staircase, 
                   momentum ):
    """Set up training ops"""

    with tf.name_scope( "train" ):

        if tune_scope:
            scope=tune_scope
        else:            
            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope )        

        loss = model.ctc_loss_layer( rnn_logits,label,sequence_length ) 

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )

        with tf.control_dependencies( extra_update_ops ):
            
            # Calculate the learning rate given the parameters
            learning_rate_tensor = tf.train.exponential_decay(
                learning_rate,
                tf.train.get_global_step(),
                decay_steps,
                decay_rate,
                staircase=decay_staircase,
                name='learning_rate' )

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate_tensor,
                beta1=momentum )

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate_tensor, 
                optimizer=optimizer,
                variables=rnn_vars )

            tf.summary.scalar( 'learning_rate', learning_rate_tensor )

    return train_op, loss




def _get_testing( rnn_logits,sequence_length,label,label_length,
                  continuous_eval, lexicon, lexicon_prior ):
    """Create ops for testing (all scalars): 
       loss: CTC loss function value, 
       label_error:   batch level edit distance on beam search max
       sequence_error: batch level sequence error rate
    """

    with tf.name_scope( "train" ):
        # Reduce by mean (rather than sum) if doing continuous evaluation
        batch_loss = model.ctc_loss_layer( rnn_logits,label,sequence_length,
                                           reduce_mean=continuous_eval) 
    with tf.name_scope( "test" ):
        predictions,_ = _get_output( rnn_logits, sequence_length,
                                     lexicon, lexicon_prior )

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
        
    return batch_loss, batch_num_label_errors, batch_num_sequence_errors, \
        batch_num_labels, predictions


def _get_loss_ops( batch_loss ):
    """Calculates the total loss by accumulating for batches and returns
    the average"""

    var_collections=[tf.GraphKeys.LOCAL_VARIABLES]

    # Variable to tally across batches (all initially zero)
    total_loss = tf.Variable( 0, trainable=False,
                              name='total_loss',
                              dtype=tf.float32,
                              collections=var_collections )

    # Create the "+=" update op
    update_op = tf.assign_add( total_loss, batch_loss )

    return total_loss, update_op


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

def _get_lexicon_output( rnn_logits, sequence_length, lexicon ):
    """Create lexicon-restricted output ops
        prediction: Dense BxT tensor of predicted character indices
        seq_prob: Bx1 tensor of output sequence probabilities
    """
    # Note: TFWordBeamSearch.so must be in LD_LIBRARY_PATH (on *nix)
    # from github.com/weinman/CTCWordBeamSearch branch var_seq_len
    word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')
    beam_width = _ctc_beam_width
    with open(lexicon) as lexicon_fd:
        corpus = lexicon_fd.read().encode('utf8')

    rnn_probs = tf.nn.softmax(rnn_logits, axis=2) # decodes in expspace

    # CTCWordBeamSearch requires a non-word char. We hack this by
    # prepending a zero-prob " " entry to the rnn_probs
    rnn_probs = tf.pad( rnn_probs,
                        [[0,0],[0,0],[1,0]], # Add one slice of zeros
                        mode='CONSTANT',
                        constant_values=0.0 )
    chars = (' '+charset.out_charset).encode('utf8')

    # Assume words can be formed from all chars--if punctuation is added
    # or numbers (etc) are to be treated differently, more such 
    # categories should be added to the charset module
    wordChars = chars[1:]
            
    prediction,seq_prob = word_beam_search_module.word_beam_search(
        rnn_probs,
        sequence_length,
        beam_width,
        'Words', # Use No LM
        0.0, # Irrelevant: No LM to smooth
        corpus, # aka lexicon [are unigrams ignored?]
        chars,
        wordChars )
    prediction = prediction - 1 # Remove hacky prepended non-word char

    return prediction, seq_prob


def _get_open_output( rnn_logits, sequence_length ):
    """Create open-vocabulary output ops for validation (testing) and prediction
       prediction: BxT sparse result of CTC beam search decoding
       seq_prob: Score of prediction
    """
    prediction,log_prob = tf.nn.ctc_beam_search_decoder(
        rnn_logits,
        sequence_length,
        beam_width=_ctc_beam_width,
        top_paths=1,
        merge_repeated=True )
    seq_prob = tf.math.exp(log_prob)

    return prediction, seq_prob


def _get_merged_output( lex_prediction, lex_seq_prob,
                        open_prediction, open_seq_prob, lexicon_prior ):
    """Create merged output ops based on maximum posterior probobability
    """
    # TODO: See whether tf.where with x and y would be faster/easier
    # Calculate posterior probability for lexicon versus open prediction
    seq_joint = tf.concat( [lexicon_prior * lex_seq_prob,
                            (1-lexicon_prior) * open_seq_prob ],
                           axis=1 )  # Bx2
    #seq_post = seq_joint / tf.reduce_sum( seq_joint, axis=1, keepdims=True)
    # argmax posterior to find most likely prediction
    seq_class = tf.argmax( seq_joint, axis=1, output_type=tf.int32 )
    # stack predictions for gathering
    predictions = tf.stack( [lex_prediction, open_prediction], axis=0) # 2xBxT
    # pair off classification (first index) and batch element [0,B) for gather
    indices = tf.stack( [seq_class,
                         tf.range( tf.shape(seq_class)[0]) ],
                        axis=1) # Bx2
    prediction = tf.gather_nd( predictions, indices) # BxT
    seq_prob = tf.gather_nd( tf.transpose(seq_joint), indices) # Bx1

    return prediction, seq_prob

def _get_output( rnn_logits, sequence_length, lexicon, lexicon_prior ):
    """Create output ops for validation (testing) and prediction
       prediction: Result of CTC beam search decoding
       seq_prob: Score of prediction
    """
    with tf.name_scope("test"):
        if lexicon:
            ctc_blank = (rnn_logits.shape[2]-1)
            lex_prediction,lex_seq_prob = _get_lexicon_output(rnn_logits,
                                                      sequence_length, lexicon )
            if lexicon_prior != None:
                # Need to run both open and closed vocabulary modes
                open_prediction, open_seq_prob = _get_open_output(
                    rnn_logits, sequence_length)
                # Convert top open output prediction to dense values
                # NOTE: What to do if the sparse result is shorter than T?
                # Reshape sparse version of open_prediction?
                open_prediction = tf.cast(
                    tf.sparse.to_dense(
                        tf.sparse.reset_shape(
                            open_prediction[0],
                            new_shape=tf.shape(lex_prediction) ),
                        default_value=ctc_blank),
                    tf.int32)
                prediction, seq_prob = _get_merged_output(
                    lex_prediction, lex_seq_prob,
                    open_prediction, open_seq_prob, lexicon_prior )
            else:
                prediction = lex_prediction
                seq_prob = lex_seq_prob
                
            # Match tf.nn.ctc_beam_search_decoder outputs: list of sparse

            # (1) CTCWordBeamSearch returns a dense tensor matching input 
            # sequence length (padded with ctc blanks).
            # We convert to sparse tightly so trailing blanks are trimmed from 
            # the dense_shape of the resulting SparseTensor
            prediction = utils.dense_to_sparse_tight(
                prediction,
                eos_token=ctc_blank )
            # (2) CTCWordBeamSearch returns only top match, so convert to list
            prediction = [prediction]
        else:
            prediction, seq_prob = _get_open_output(rnn_logits, sequence_length)
            
    return prediction, seq_prob


def train_fn( scope, tune_from, learning_rate, 
                    decay_steps, decay_rate, decay_staircase, momentum ): 
    """Returns a function that trains the model"""

    def train( features, labels, mode ):

        logits, sequence_length = _get_image_info( features, mode )

        train_op, loss = _get_training( logits,labels,
                                        sequence_length, 
                                        scope, learning_rate, 
                                        decay_steps, decay_rate, 
                                        decay_staircase, momentum )
        
        # Initialize weights from a pre-trained model
        # NOTE: Does not work when num_gpus>1, cf. tensorflow issue 21615.
        scaffold = tf.train.Scaffold( init_fn=
                                      _get_init_pretrained( tune_from ) )

        return tf.estimator.EstimatorSpec( mode=mode, 
                                           loss=loss, 
                                           train_op=train_op,
                                           scaffold=scaffold )
    return train


def evaluate_fn( lexicon=None, lexicon_prior=None ):
    """Returns a function that evaluates the model for all batches at once or 
    continuously for one batch"""

    def evaluate( features, labels, mode, params ):
                
        logits, sequence_length = _get_image_info( features, mode )

        continuous_eval = params['continuous_eval']
        length = features['length']
            
        # Get the predictions
        batch_loss,\
            batch_label_error,\
            batch_sequence_error, \
            batch_total_labels, \
            _ = _get_testing( logits,sequence_length,labels, length, 
                              continuous_eval, lexicon, lexicon_prior )
        
        # Label errors: mean over the batch and updated total number
        mean_label_error, \
            update_op_label, \
            total_num_label_errors, \
            total_num_labels = _get_label_err_ops( batch_label_error, 
                                                  batch_total_labels )
        
        # Sequence errors: mean over the batch and updated total number
        mean_sequence_error,\
            update_op_seq,\
            total_num_sequence_errs,\
            total_num_sequences = _get_seq_err_ops( batch_sequence_error, 
                                                   length )
        
        # Loss: Accumulated total loss over batches
        total_loss, update_op_loss   = _get_loss_ops( batch_loss )
        mean_loss =  tf.truediv( total_loss, 
                                 tf.cast( total_num_sequences, tf.float32 ),
                                 name='mean_loss' )    
   

        # Print the metrics while doing continuous evaluation (evaluate.py) 
        # Note: tf.Print is identical to tf.identity, except it prints
        # the list of metrics as a side effect
        if (continuous_eval):
            global_step = tf.train.get_or_create_global_step()
            mean_sequence_error = tf.Print( mean_sequence_error, 
                                            [global_step,
                                             batch_loss,
                                             mean_label_error,
                                             mean_sequence_error] ,
                                            first_n=1)
            
            # Create summaries for the metrics during continuous eval
            tf.summary.scalar( 'loss', tensor=batch_loss,
                               family='test' )
            tf.summary.scalar( 'label_error', tensor=mean_label_error,
                               family='test' )
            tf.summary.scalar( 'sequence_error',
                               tensor=mean_sequence_error,
                               family='test' )
            
        # Convert to tensor from Variable in order to pass it to eval_metric_ops
        total_num_label_errors  = tf.convert_to_tensor( total_num_label_errors )
        total_num_labels        = tf.convert_to_tensor( total_num_labels )
        total_num_sequence_errs = tf.convert_to_tensor( total_num_sequence_errs )
        total_num_sequences     = tf.convert_to_tensor( total_num_sequences )
        total_loss              = tf.convert_to_tensor( total_loss )
            
        # All the ops that will be passed to the EstimatorSpec object
        eval_metric_ops = {
            'mean_loss': ( mean_loss, update_op_loss ),
            'mean_label_error': ( mean_label_error, update_op_label ),
            'mean_sequence_error': ( mean_sequence_error, update_op_seq ),
            'total_loss': ( total_loss, tf.no_op() ),
            'total_num_label_errors': ( total_num_label_errors, tf.no_op() ),
            'total_num_labels':( total_num_labels, tf.no_op() ),
            'total_num_sequence_errs': ( total_num_sequence_errs, tf.no_op() ),
            'total_num_sequences': ( total_num_sequences, tf.no_op() ) 
        }
        
        return tf.estimator.EstimatorSpec( mode=mode, 
                                           loss=batch_loss, 
                                           eval_metric_ops=eval_metric_ops )
    return evaluate


def predict_fn( lexicon, lexicon_prior ):
    """Returns a function that runs the model on the input data 
       (e.g., for validation)"""

    def predict( features, labels, mode ):

        logits, sequence_length = _get_image_info(features, mode)
        
        predictions, log_probs = _get_output( logits, sequence_length,
                                              lexicon, lexicon_prior )

        if lexicon:
            # TFWordBeamSearch produces only a single value, but its
            # given dense shape is the original sequence length
            # dense_to_sparse_tight in_get_output should filter out
            # the excess, but we set the dense fill value to ctc_blank
            # now to catch any potential errors/bugs downstream later
            ctc_blank = (logits.shape[2]-1)
            final_pred = tf.sparse.to_dense( predictions[0],
                                             default_value=ctc_blank ) 
        else:
        # tf.nn.ctc_beam_search produces SparseTensor but EstimatorSpec
        # predictions only takes dense tensors
            final_pred = tf.sparse.to_dense( predictions[0], 
                                             default_value=0 ) 
        
        return tf.estimator.EstimatorSpec( mode=mode,
                                           predictions={ 'labels': final_pred,
                                                         'score': log_probs })

    return predict
