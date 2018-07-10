# FILTERS:
# To be used as filter_fn's passed into call to pipeline.get_data
# Filters for dynamic data must be written in the following structure:
# * filter_fn(image, width, label, length, text)
# Filters for static data must be written in the following structure:
# * filter_fn(image, width, label, length, text, filename)
# Components are structured as follows:
# image          : tf.float32, [32, ?, 1] HWC
# width          : tf.int32, scalar
# label          : tf.int64, sparse tensor, [?]
# length         : tf.int32, scalar
# text, filename : tf.string, scalar
 
import tensorflow as tf
import model

# Note: The following 2 filters only work for dynamic data bc no filename param

def dyn_filter_by_width(image, width, label, length, text):
    return tf.greater(width, 20)

def dyn_ctc_loss_filter(image, width, label, length, text):
    seq_len = model.get_sequence_lengths(width)
    return tf.greater(seq_len, 0)
