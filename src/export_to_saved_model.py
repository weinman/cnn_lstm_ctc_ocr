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

# validate.py - Run model directly on from paths to image filenames. 
# NOTE: assumes mjsynth files are given by adding an extra row of padding

import tensorflow as tf

import model_fn as model_fn
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'ckpt_path','/media/path/to/models/',
                            """Directory for model checkpoints""" )
tf.app.flags.DEFINE_string( 'checkpoint','model.ckpt-73152',
                            """The checkpoint to export. Just the name of the checkpoint, 
                            no path needed. Exemple: 'model.ckpt-104976'""" )
tf.app.flags.DEFINE_string( 'export_dir','/media/path/to/where/to/export/saved_model/',
                            """Path to the directory where SavedModel is to be exported""" )
tf.app.flags.DEFINE_string( 'lexicon','',
                            """File containing lexicon of image words""" )
tf.app.flags.DEFINE_float( 'lexicon_prior',None,
                           """Prior bias [0,1] for lexicon word""" )

IMAGE = 'image'
WIDTH = 'width'
LABELS = 'labels'
LENGTH = 'length'
TEXT = 'text'


class ExportSavedModel():

    def __get_config(self):
        """Setup session config to soften device placement"""

        device_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False )

        custom_config = tf.estimator.RunConfig( session_config=device_config )

        return custom_config

    def __serving_input_receiver(self):
        """Placeholders function required by the Estimator to export SavedModel"""

        image = tf.placeholder(dtype=tf.float32, shape=[1, 32, None, 1], name=IMAGE)
        width = tf.placeholder(dtype=tf.int32, shape=[], name=WIDTH)
        labels = tf.placeholder(dtype=tf.float32, shape=[None], name=LABELS)
        length = tf.placeholder(dtype=tf.float32, shape=[], name=LENGTH)
        text = tf.placeholder(dtype=tf.string, shape=[], name=TEXT)
        features = {IMAGE: image,
                    WIDTH: width,
                    LABELS: labels,
                    LENGTH: length,
                    TEXT: text}
        receiver_tensor = {IMAGE: image,
                           WIDTH: width}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

    def export_saved_model(self):
        """Will restore the given checkpoint and export it into the SavedModel format"""

        classifier = tf.estimator.Estimator(config=self.__get_config(),
                                            model_fn=model_fn.predict_fn(
                                                 FLAGS.lexicon,
                                                 FLAGS.lexicon_prior),
                                            model_dir=FLAGS.ckpt_path)

        checkpoint = os.path.join(FLAGS.ckpt_path, FLAGS.checkpoint)

        classifier.export_saved_model(FLAGS.export_dir,
                                      serving_input_receiver_fn=self.__serving_input_receiver,
                                      checkpoint_path=checkpoint)

if __name__ == '__main__':
    export = ExportSavedModel()
    export.export_saved_model()
