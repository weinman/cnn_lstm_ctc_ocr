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

import mjsynth
import model


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

tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer='Adam'
mode = learn.ModeKeys.TRAIN # 'Configure' training mode for dropout layers

def _get_input():
    """Set up and return image, label, and image width tensors"""

    image,width,label,_,_,_=mjsynth.bucketed_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        input_device=FLAGS.input_device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold )

    #tf.summary.image('images',image) # Uncomment to see images in TensorBoard
    return image,width,label

def _get_single_input():
    """Set up and return image, label, and width tensors"""

    image,width,label,length,text,filename=mjsynth.threaded_input_pipeline(
        deps.get('records'), 
        str.split(FLAGS.filename_pattern,','),
        batch_size=1,
        num_threads=FLAGS.num_input_threads,
        num_epochs=1,
        batch_device=FLAGS.input_device, 
        preprocess_device=FLAGS.input_device )
    return image,width,label,length,text,filename

def _get_training(rnn_logits,label,sequence_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope="convnet|rnn"

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

    return train_op

def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not FLAGS.tune_from:
        return None
    
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ckpt_path=FLAGS.tune_from

    init_fn = lambda sess: saver_reader.restore(sess, ckpt_path)

    return init_fn


def main(argv=None):

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        
        image, width, label = _get_input()

        with tf.device(FLAGS.train_device):
            features, sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            train_op = _get_training(logits,label,sequence_length)

        session_config = _get_session_config()

        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        # There are three hooks to maintain
        #   summary
        #   checkpoint
        #   model
        # I believe the verbage is saying that model == checkpoint, so I am seeing if checkpoint_hook
        #   has all the functionality we need to reproduce supervisor in conjunction with summary_hook

        init_scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_fn=_get_init_pretrained()
        )
        
        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.output,
            save_secs=150
        )

        summary_hook = tf.train.SummarySaverHook(
            output_dir=FLAGS.output,
            save_secs=30,
            summary_op=summary_op
        )
        
        monitor = tf.train.MonitoredTrainingSession(
            hooks=[saver_hook, summary_hook],
            config=session_config,
            scaffold=init_scaffold
        )
        
        with monitor as sess:
            step = sess.run(global_step)
            while step < FLAGS.max_num_steps:
                if monitor.should_stop():
                    break
                [step_loss, step]=sess.run([train_op, global_step])
            monitor.saver.save( sess, os.path.join(FLAGS.output, 'model.ckpt'),
                                global_step=global_step)

        """
        sv = tf.train.Supervisor(
            logdir=FLAGS.output,             # Optional path to a directory where to checkpoint the model and log events for the visualizer. Used by chief supervisors.
            init_op=init_op,                 # Used by chief supervisors to initialize the model when it can not be recovered. Defaults to an Operation that initializes all global variables.
            summary_op=summary_op,           # An Operation that returns a Summary for the event logs. Used by chief supervisors if a logdir was specified.
            save_summaries_secs=30,          # Number of seconds between the computation of summaries for the event log. 
            init_fn=_get_init_pretrained(),  # Optional callable used to initialize the model. Called after the optional init_op is called. The callable must accept one argument, the session being initialized.
            save_model_secs=150)             # Number of seconds between the creation of model checkpoints.

        with sv.managed_session(config=session_config) as sess:
            step = sess.run(global_step)
            while step < FLAGS.max_num_steps:
                if sv.should_stop():
                    break                    
                [step_loss,step]=sess.run([train_op,global_step])
            sv.saver.save( sess, os.path.join(FLAGS.output,'model.ckpt'),
                           global_step=global_step)
        """

if __name__ == '__main__':
    tf.app.run()
