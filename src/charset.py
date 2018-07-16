import tensorflow as tf
import numpy as np

# The list (well, string) of valid output characters
# If any example contains a character not found here, 
# you'll get a runtime error when this is encountered.
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

""" 
Dict for constant time string->label conversion
Attribution: https://stackoverflow.com/questions/36459969/python-convert-list-to-dictionary-with-indexes -- from user: Abhijit
Produces a table of character->index mappings according to out_charset
"""
out_charset_dict = { key: val for val, key in enumerate( out_charset ) }

def num_classes():
    """ Returns length/size of out_charset """
    return len( out_charset )

# get_string and get_strings courtesy of Jerod Weinman. See:
# https://github.com/weinman/cnn_lstm_ctc_ocr/blob/f98902564ed9883f00267557ac8e386771fab7aa/src/mjsynth.py#L29
# get_string tweaked to use dict rather than index member func
def get_strings( labels ):
    """
    Transform a SparseTensorValue matrix of labels into the corresponding
    list of character strings
    """
    num_strings = labels.dense_shape[0]
    
    strings = []

    for row in range( num_strings ):

        indices = np.where( labels.indices[:,0]==row )[0]
        indices.shape = ( indices.shape[0], 1 )

        label = labels.values[indices]
        label.shape = ( label.shape[0] )

        strings.append( get_string( label ) )

    return strings

def get_string( labels ):
    """
    Transform an 1D array of labels into the corresponding character string
    """
    string = ''.join( [out_charset_dict[c] for c in labels] )
    return string


