import tensorflow as tf
import numpy as np

# The list (well, string) of valid output characters
# If any example contains a character not found here, 
# you'll get a runtime error when this is encountered.
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_num_classes = len(out_charset)

def get_data(static,
             base_dir=None,
             file_patterns=None,
             num_threads=4,
             batch_size=32,
             boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
             input_device=None,
             num_epoch=None,
             filter_fn=None):
    
    # Elements to be buffered
    num_buffered_elements = num_threads*batch_size*2

    # Get correct import and args for given pipeline
    if static:
        import mjsynth as dpipe
        args = (base_dir, file_patterns, num_threads, num_buffered_elements)
    else:
        import maptextsynth as dpipe
        args = None # Place future args for synthetic data here...

    with tf.device(input_device):
        # Get raw data
        dataset = dpipe.get_dataset(args).prefetch(num_buffered_elements)
        
        # Preprocess data
        dataset = dataset.map(dpipe.preprocess_fn, 
                              num_parallel_calls=num_threads)
        dataset = dataset.prefetch(num_buffered_elements)

        # Remove input that doesn't fit necessary specifications
        if filter_fn:
            dataset = dataset.filter(filter_fn).prefetch(num_buffered_elements)

        # Bucket and batch appropriately
        if boundaries:
            dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                element_length_func=dpipe.element_length_fn,
                bucket_batch_sizes=np.full(len(boundaries)+1, batch_size),
                bucket_boundaries=boundaries,)) 
        else:
            # Dynamically pad batches to match largest in batch
            dataset = dataset.padded_batch(batch_size, 
                                           padded_shapes=dataset.output_shapes,)

        # Update to account for batching
        num_buffered_elements = num_threads*2
        
        dataset = dataset.prefetch(num_buffered_elements)
        
        # Repeat for num_epochs  
        if num_epoch and static:
            dataset = dataset.repeat(num_epoch)

        # Convert labels to sparse tensor for CNN purposes
        dataset = dataset.map(dpipe.postbatch_fn,
                              num_parallel_calls=num_threads)
        dataset = dataset.prefetch(num_buffered_elements)
        
    return dataset

def num_classes():
    return _num_classes
