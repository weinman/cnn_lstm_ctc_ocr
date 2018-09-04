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
    """Read lexicon entries from a file (one per line), adding initial
    capitalization and all uppercase transformations to each entry

    """
    with open(fname, 'r') as fd:
        # for each line, return line, init_caps, all_caps
        # Ex: line="  abc " -> ("abc", "Abc", "ABC")
        vocab = list(chain.from_iterable((line.strip(), line.strip().title(), 
				          line.strip().upper()) for line in fd))
    return vocab 


def dictionary_from_file(fname, charset):
    """Create a label-indexed version of the lexicon from a lexicon file name.
    Parameters:
      fname   : path to the file name containing the lexicon entries
      charset : string containing one instance of each valid character
    Returns:
      tensor_dict : A tf.SparseTensor with one row per lexicon entry and 
                    columns containing indices of corresponding chracters
                    in charset.
    """
    vocab = read_dict(fname)
    return dictionary_from_list(vocab, charset)


def dictionary_from_list(vocab, charset):
    """Create a label-indexed version of the lexicon from a list of strings.
    Parameters:
      vocab   : list of strings in the lexicon
      charset : string containing one instance of each valid character
    Returns:
      tensor_dict : A tf.SparseTensor with one row per lexicon entry and 
                    columns containing indices of corresponding chracters
                    in charset.
    """

    # inds are locations of valid character values
    inds = np.array(
            [[i, j] for i,word in enumerate(vocab) for j in range(len(word))],
            dtype=np.int32)
    # parse each character label using charset as index reference
    vals = np.array(
            [charset.index(ch) for word in vocab for ch in word], 
	    dtype=np.int32)
    dims = np.array(
            [len(vocab), max(map(lambda x: len(x), vocab))], dtype=np.int32)
    tensor = tf.SparseTensorValue(indices=inds, values=vals, dense_shape=dims)
    tensor = tf.convert_to_tensor_or_sparse_tensor(tensor)
    return tensor
