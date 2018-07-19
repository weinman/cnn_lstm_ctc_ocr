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



# READ THIS BEFORE WRITING YOUR OWN FILTER FUNCTION
#
# To be used as filter_fn's passed into call to pipeline.get_data
# Filters for dynamic data must be written in the following structure:
# * filter_fn(image, width, label, length, text)
# VAR            : tf.dtype, shape
#--------------------------------------------
# image          : tf.float32, [32, ?, 1] HWC
# width          : tf.int32, []
# label          : tf.int32, [?]
# length         : tf.int32, []
# text           : tf.string, []
#
# Filters for static data must be written in the following structure:
# * filter_fn(image, width, label, length, text, filename)
# VAR            : tf.dtype, shape
#--------------------------------------------
# image          : tf.float32, [32, ?, 1] HWC
# width          : tf.int32, []
# label          : tf.int64, SparseTensor [?]
# length         : tf.int64, []
# text, filename : tf.string, []
#
# Note: For more detailed information, refer to `_preprocess_fn`
# in mjsynth.py and maptextsynth.py. Filtering is performed after
# these transformations have been applied. Therefore, filter_fn args
# will correspond to the return values of `preprocess_fn`. 
 
import tensorflow as tf
import model

# Note: The following 2 filters only work for dynamic data. No filename param.

def dyn_filter_by_width( image, width, label, length, text ):
    return tf.greater( width, 20 )

def dyn_ctc_loss_filter( image, width, label, length, text ):
    seq_len = model.get_sequence_lengths( width )
    return tf.greater( seq_len, 0 )
