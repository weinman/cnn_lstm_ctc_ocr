# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Matthew Murphy
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

# lexicon.py -- A suite of tools for loading a label lexicon into
#   SparseTensorValue entries from a file of words (for use with a
#   lexicon-restricted CTC decoder).

from itertools import chain
import numpy as np
import tensorflow as tf

def read_dict(fname):
    f = open(fname, 'r')
    # for each line, return line, init_caps, all_caps
    # Ex: line="  abc " -> ("abc", "Abc", "ABC")
    vocab = list(chain.from_iterable((line.strip(), line.strip().title(), 
				 line.strip().upper()) for line in f))
    return vocab 

def dictionary_from_file(fname, char_string):
    vocab = read_dict(fname)
    return dictionary_from_list(vocab, char_string)
   
def dictionary_from_list(vocab, char_string):
    # inds are all non-zero char values
    inds = np.array(
            [[i, j] for i,word in enumerate(vocab) for j in range(len(word))],
            dtype=np.int32)
    # parse each character label using char_string as index reference
    vals = np.array(
            [char_string.index(c) for word in vocab for c in word], 
	    dtype=np.int32)
    dims = np.array(
            [len(vocab), max(map(lambda x: len(x), vocab))], dtype=np.int32)
    tensor = tf.SparseTensorValue(indices=inds, values=vals, dense_shape=dims)
    tensor = tf.convert_to_tensor_or_sparse_tensor(tensor)
    return tensor
