import tensorflow as tf
import numpy as np

def get_data( static_data,
              base_dir=None,
              file_patterns=None,
              num_threads=4,
              batch_size=32,
              boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
              input_device=None,
              num_epochs=None,
              filter_fn=None ):
    """Get dataset according to parameters
    Parameters:
      static_data   : boolean for whether to use static or dynamic data
      base_dir      : string for static data locations (static data only)
      file_patterns : string for static data patterns  (static data only)
      num_threads   : number of threads to use for IO / preprocessing
      batch_size    : batch size
      boundaries    : boundaries for bucketing. If None, no bucketing
      input_device  : Device for pinning ops
      num_epochs    : if None, data repeats infinitely (static data only)
      filter_fn     : filtering function
    Returns:
      dataset : elements structured as [features, labels]
                feature structure can be seen in postbatch_fn 
                in mjsynth.py or maptextsynth.py for static or dynamic
                data pipelines respectively
    """    
    # Elements to be buffered
    num_buffered_elements = num_threads * batch_size * 2

    # Get correct import and args for given pipeline
    # `dpipe` will be a variable for the package name
    if static_data:
        import mjsynth as dpipe
        dpipe_args = ( base_dir, 
                       file_patterns, 
                       num_threads, 
                       num_buffered_elements )
    else:
        import maptextsynth as dpipe
        dpipe_args = None # Place future args for synthetic data here...

    with tf.device( input_device ):
        # Get raw data
        dataset = dpipe.get_dataset( dpipe_args )
        dataset = dataset.prefetch( num_buffered_elements )
        
        # Preprocess data
        dataset = dataset.map( dpipe.preprocess_fn, 
                               num_parallel_calls=num_threads )
        dataset = dataset.prefetch( num_buffered_elements )

        # Remove input that doesn't fit necessary specifications
        if filter_fn:
            dataset = dataset.filter( filter_fn )
            dataset = dataset.prefetch( num_buffered_elements )

        # Bucket and batch appropriately
        if boundaries:
            dataset = dataset.apply( tf.contrib.data.bucket_by_sequence_length(
                element_length_func=dpipe.element_length_fn,
                # Create numpy array as follows: [batch_size,...,batch_size]
                bucket_batch_sizes=np.full( len( boundaries ) + 1, 
                                            batch_size ),
                bucket_boundaries=boundaries ) ) 
        else:
            # Dynamically pad batches to match largest in batch
            dataset = dataset.padded_batch( batch_size, 
                                            padded_shapes=dataset.output_shapes )

        # Update to account for batching
        num_buffered_elements = num_threads * 2
        
        dataset = dataset.prefetch( num_buffered_elements )
        
        # Repeat for num_epochs  
        if num_epochs and static_data:
            dataset = dataset.repeat( num_epoch )
        # Repeat indefinitely if no num_epochs is specified
        elif static_data:
            dataset = dataset.repeat()

        # Prepare dataset for Estimator ingestion
        # ie: sparsify labels for CTC operations (eg loss, decoder)
        # and convert elements to be [features, label]
        dataset = dataset.map( dpipe.postbatch_fn,
                               num_parallel_calls=num_threads )
        dataset = dataset.prefetch( num_buffered_elements )
        
    return dataset

def rescale_image( image ):
    """Rescale from uint8([0,255]) to float([-0.5,0.5])"""
    image = tf.image.convert_image_dtype( image, tf.float32 )
    image = tf.subtract( image, 0.5 )
    return image
