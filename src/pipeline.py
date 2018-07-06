import tensorflow as tf

out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_num_classes = len(out_charset)

def get_static_data(base_dir,
                    file_patterns,
                    num_threads=4,
                    batch_size=32,
                    boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
                    input_device=None,
                    num_epoch=None,
                    filter_fn=None):
    import mjsynth
    return mjsynth.get_data(base_dir=base_dir, 
                            file_patterns=file_patterns,
                            num_threads=num_threads, 
                            batch_size=batch_size, 
                            boundaries=boundaries, 
                            input_device=input_device,
                            num_epoch=num_epoch, 
                            filter_fn=filter_fn)
    
def get_dynamic_data(num_threads=4,
                     batch_size=32,
                     boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
                     input_device=None,
                     filter_fn=None):
    import dynmj
    return dynmj.get_data(num_threads=num_threads, 
                          batch_size=batch_size, 
                          boundaries=boundaries, 
                          input_device=input_device, 
                          filter_fn=filter_fn)

def num_classes():
    return _num_classes
