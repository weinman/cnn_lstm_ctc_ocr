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
