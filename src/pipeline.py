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

# pipeline.py -- Constructs Dataset objects for use in training, testing, etc.

import tensorflow as tf
import numpy as np

def get_data( static_data,
              base_dir=None,
              file_patterns=None,
              num_threads=4,
              batch_size=32,
              boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
              num_epochs=None,
              filter_fn=None,
              synth_config_file=None,
              synth_lexicon_file=None ):
    """Get Dataset according to parameters
    Parameters:
      static_data   : boolean for whether to use static or dynamic data
      base_dir      : string for static data locations (static data only)
      file_patterns : string for static data patterns  (static data only)
      num_threads   : number of threads to use for IO / preprocessing
      batch_size    : batch size
      boundaries    : boundaries for bucketing. If None, no bucketing
      num_epochs    : if None, data repeats infinitely (static data only)
      filter_fn     : filtering function
      synth_config_file: 
                      string for synthesizer config file (dynamic data only)
      synth_lexicon_file: 
                      string for synthesizer lexicon file (dynamic data only)
    Returns:
      dataset : tf.data.Dataset object.
                elements structured as [features, labels]
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
        # This is for dynamic data only -- refer to README.md
        # for more usage instructions if relevant
        import maptextsynth as dpipe
        dpipe_args = ( synth_config_file,
                       synth_lexicon_file )

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
        dataset = dataset.repeat( num_epochs )
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
