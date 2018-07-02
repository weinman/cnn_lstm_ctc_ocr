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

#import dynmj
import mjsynth
import model
import model_fn

FLAGS = tf.app.flags.FLAGS

# For displaying various statistics while training
tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer='Adam'

def _get_input_stream():
    """Set up and return image, label, and image width tensors"""

    dataset=mjsynth.bucketed_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern_training,','),
        batch_size=FLAGS.training_batch_size,
        num_threads=FLAGS.num_input_threads_training,
        input_device=FLAGS.input_device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold)

    iterator = dataset.make_one_shot_iterator() 

    image, width, label, _, _, _ = iterator.get_next()

    # The input for the model function 
    features = {"image": image, "width": width, "optimizer": optimizer}
    
    return features, label

def _get_single_input_stream():    
    """Set up and return image, label, and width tensors"""

    dataset=dynmj.threaded_input_pipeline(
        batch_size=1,
        num_threads=FLAGS.num_input_threads,
        num_epochs=1,
        batch_device=FLAGS.input_device, 
        preprocess_device=FLAGS.input_device )

    return dataset.make_one_shot_iterator()


def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config


def main(argv=None):  

    custom_config = tf.estimator.RunConfig(session_config=_get_session_config(),
                                           save_checkpoints_secs=30)
    
    # Initialize the classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn.model_fn, 
                                        model_dir=FLAGS.train_output,
                                        config=custom_config)

    # Train the model
    classifier.train(input_fn=lambda: _get_input_stream())

if __name__ == '__main__':
    tf.app.run()
