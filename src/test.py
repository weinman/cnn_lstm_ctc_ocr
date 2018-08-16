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

# test.py -- Calculates evaluation metrics on an entire Dataset

import tensorflow as tf
import pipeline
import filters
import model_fn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for model checkpoints""" )

tf.app.flags.DEFINE_integer( 'batch_size',2**9,
                             """Eval batch size""" )
tf.app.flags.DEFINE_string( 'test_path','../data/',
                            """Base directory for test/validation data""" )

tf.app.flags.DEFINE_string( 'filename_pattern','test/words-*',
                            """File pattern for test input data""" )
tf.app.flags.DEFINE_integer( 'num_input_threads',4,
                             """Number of readers for input data""" )
tf.app.flags.DEFINE_boolean( 'static_data', True,
                            """Whether to use static data 
                            (false for dynamic data)""" )

tf.app.flags.DEFINE_integer('min_image_width',None,
                            """Minimum allowable input image width""")
tf.app.flags.DEFINE_integer('max_image_width',None,
                            """Maximum allowable input image width""")
tf.app.flags.DEFINE_integer('min_string_length',None,
                            """Minimum allowable input string length""")
tf.app.flags.DEFINE_integer('max_string_length',None,
                            """Maximum allowable input string_length""")

tf.app.flags.DEFINE_string('synth_config_file','../data/maptextsynth_config.txt',
                           """Location of config file for map text synthesizer""")
tf.app.flags.DEFINE_string('synth_lexicon_file','../data/lexicon.txt',
                           """Location of synth lexicon""")


def _get_input():
    """
    Get tf.data.Dataset object according to command-line flags for testing
    using tf.estimator.Estimator
    Returns:
      dataset : elements structured as [features, labels]
                feature structure can be seen in postbatch_fn 
                in mjsynth.py or maptextsynth.py for static or dynamic
                data pipelines respectively
    """

    # WARNING: More than two filters causes SEVERE throughput slowdown
    filter_fn = filters.input_filter_fn \
                ( min_image_width=FLAGS.min_image_width,
                  max_image_width=FLAGS.max_image_width,
                  min_string_length=FLAGS.min_string_length,
                  max_string_length=FLAGS.max_string_length )

    # Get data according to flags
    dataset = pipeline.get_data( FLAGS.static_data,
                                 base_dir=FLAGS.test_path,
                                 file_patterns=str.split(
                                     FLAGS.filename_pattern,
                                     ','),
                                 num_threads=FLAGS.num_input_threads,
                                 batch_size=FLAGS.batch_size,
                                 filter_fn=filter_fn,
                                 synth_config_file=FLAGS.synth_config_file,
                                 synth_lexicon_file=FLAGS.synth_lexicon_file,
                                 num_epochs=1 )
    return dataset


def _get_config():
    """Setup config to soften device placement"""

    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False )

    custom_config = tf.estimator.RunConfig(session_config=device_config)

    return custom_config


def main(argv=None):
  
    # Initialize the classifier
    classifier = tf.estimator.Estimator( config = _get_config(),
                                         model_fn=model_fn.evaluate_fn(), 
                                         model_dir=FLAGS.model,
                                         params={'continuous_eval': False} )
    
    evaluations = classifier.evaluate( input_fn=_get_input )
    
    print(evaluations)

if __name__ == '__main__':
    tf.app.run()
