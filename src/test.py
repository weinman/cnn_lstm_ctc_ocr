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
import cv2
import mjsynth
import model
import model_fn

FLAGS = tf.app.flags.FLAGS

optimizer = 'Adam'

def _get_input_stream():
    """Set up and return image, label, width and text tensors"""

    dataset=mjsynth.threaded_input_pipeline(
        FLAGS.test_path, 
        str.split(FLAGS.filename_pattern_testing,','),
        batch_size=FLAGS.testing_batch_size,
        num_threads=FLAGS.num_input_threads_testing,
        batch_device=FLAGS.device,
        preprocess_device=FLAGS.device)

    iterator = dataset.make_one_shot_iterator() 
    
    image, width, label, length, _, _ = iterator.get_next()

    # The input for the model function 
    features = {"image": image, 
                "width": width, 
                "length": length, 
                "label": label, 
                "optimizer": optimizer}

    return features, label

def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config


def main(argv=None):

    custom_config = tf.estimator.RunConfig(session_config=_get_session_config())

    # Initialize the classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn.model_fn, 
                                        model_dir=FLAGS.model,
                                        config=custom_config)
    
    while True:
        evaluations = classifier.evaluate(input_fn=lambda: _get_input_stream())
        print(evaluations)

if __name__ == '__main__':
    tf.app.run()
