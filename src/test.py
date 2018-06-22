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
import time
import tensorflow as tf
from tensorflow.contrib import learn

import mjsynth
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('output','test',
                          """Sub-directory of model for test summary events""")

tf.app.flags.DEFINE_integer('batch_size',2**8,
                            """Eval batch size""")
tf.app.flags.DEFINE_integer('test_interval_secs', 60,
                             'Time between test runs')

tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")

tf.app.flags.DEFINE_string('test_path','../data/',
                           """Base directory for test/validation data""")
tf.app.flags.DEFINE_string('filename_pattern','val/words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4,
                          """Number of readers for input data""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers


def _get_input_stream():
    """Set up and return image, label, width and text tensors"""

    dataset=mjsynth.threaded_input_pipeline(
        FLAGS.test_path,
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        batch_device=FLAGS.device, 
        preprocess_device=FLAGS.device )
    
    return dataset.make_one_shot_iterator()

def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

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

def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn

def main(argv=None):
    input_stream = _get_input_stream()
    with tf.Graph().as_default():
        image,width,label,length,_ = input_stream.get_next()

        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            loss,label_error,sequence_error = _get_testing(
                logits,sequence_length,label,length)

        global_step = tf.train.get_or_create_global_step()

        session_config = _get_session_config()
        restore_model = _get_init_trained()

        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        summary_writer = tf.summary.FileWriter( os.path.join(FLAGS.model,
                                                            FLAGS.output) )

        step_ops = [global_step, loss, label_error, sequence_error]

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)
            summary_writer.add_graph(sess.graph)

            try:            
                while True:
                    # Get latest checkpoint
                    restore_model(sess, _get_checkpoint()) 
                    
                    step_vals = sess.run(step_ops)
                    print step_vals
                    
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step_vals[0])

            except tf.errors.OutOfRangeError:
                print('Done')

if __name__ == '__main__':
    tf.app.run()
