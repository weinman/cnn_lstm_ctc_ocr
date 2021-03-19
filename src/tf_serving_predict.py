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

import numpy as np
from PIL import Image

from tensorflow import saved_model as sm

import charset as charset

import datetime

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

global ctc_graph
ctc_graph = tf.compat.v1.get_default_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'server_host','localhost:8500',
                            """IP:PORT of the Tensorflow Model Serving""" )
tf.app.flags.DEFINE_string( 'model_spec_name','crnn',
                            """Name of the saved models batch as it is referenced in the "tf_server.conf""" )
tf.app.flags.DEFINE_integer( 'model_version',1616119579,
                            """SavedModel version to call""" )
tf.app.flags.DEFINE_string( 'image_path','/media/test/images/test_image.png',
                            """Image to be used as input for inference/prediction""" )
tf.app.flags.DEFINE_integer( 'image_base_height',32,
                            """The height for all tensors to be reshaped to""" )
tf.app.flags.DEFINE_string( 'lexicon','',
			                """File containing lexicon of image words""" )
tf.app.flags.DEFINE_float( 'lexicon_prior',None,
			                """Prior bias [0,1] for lexicon word""" )

tf.logging.set_verbosity( tf.logging.INFO )

server = FLAGS.server_host
channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.signature_name = sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
request.model_spec.name = FLAGS.model_spec_name
request.model_spec.version.value = FLAGS.model_version

IMAGE = 'image'
WIDTH = 'width'
LABELS = 'labels'

class Predict():

    def __get_image(self, filename):
        """Load image data for placement in graph"""

        pil_image = Image.open(filename)
        width, height = pil_image.size
        new_width = FLAGS.image_base_height * width / height
        pil_image = pil_image.resize((int(new_width), FLAGS.image_base_height), Image.ANTIALIAS)
        image = np.array(pil_image)

        # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
        image = image[:, :, :1]  # so just extract first channel, preserving 3D shape

        return image

    def predict(self):
        """ Will call Tensorflow Model Server for Prediction, using gRPC API. """

        print("Prediction Start Time: ", datetime.datetime.now())
        image = self.__get_image(FLAGS.image_path)

        h, w, c = image.shape
        image = np.asarray(image).astype(np.float32)
        image = image[np.newaxis, :, :, :]

        request.inputs[IMAGE].CopyFrom(tf.compat.v1.make_tensor_proto(image, dtype=None, shape=image.shape))
        request.inputs[WIDTH].CopyFrom(tf.compat.v1.make_tensor_proto(w))
        result = stub.Predict(request, 10)

        protobuf_response = result.outputs[LABELS]
        ndarray_response = tf.make_ndarray(protobuf_response)
        label = charset.label_to_string(ndarray_response[0])

        print("Prediction End Time: ", datetime.datetime.now())
        print("Predicted Text: ", label)

        return label

if __name__ == '__main__':
    pr = Predict()
    pr.predict()
