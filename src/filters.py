# FILTERS:
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

# Filters for static data must be written in the following structure:
# * filter_fn(image, width, label, length, text, filename)
# VAR            : tf.dtype, shape
#--------------------------------------------
# image          : tf.float32, [32, ?, 1] HWC
# width          : tf.int32, []
# label          : tf.int64, SparseTensor [?]
# length         : tf.int64, []
# text, filename : tf.string, []

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
