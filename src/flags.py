import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#Flags used in training
tf.app.flags.DEFINE_string('train_output','../data/model',
                           """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from','',
                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope','',
                          """Variable scope for training""")

tf.app.flags.DEFINE_integer('batch_size_train',2**5,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate',1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps',2**16,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_boolean('decay_staircase',False,
                          """Staircase learning rate decay by integer division""")


tf.app.flags.DEFINE_integer('max_num_steps', 2**21,
                            """Number of optimization steps to run""")

tf.app.flags.DEFINE_string('train_device','/gpu:1',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device','/gpu:0',
                           """Device for preprocess/batching graph placement""")
tf.app.flags.DEFINE_boolean('static_data', True,
                            """Whether to use static data 
                            (false for dynamic data)""")
tf.app.flags.DEFINE_string('train_path','../data/train/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern_train','words-*',
                           """File pattern for train input data""")
tf.app.flags.DEFINE_integer('num_input_threads_train',4,
                          """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
                            """Limit of input string length width""")


#Flags used in testing
tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")

tf.app.flags.DEFINE_integer('batch_size_eval',2**9,
                            """Eval batch size""")
tf.app.flags.DEFINE_integer('test_interval_secs', 60,
                             'Time between test runs')

tf.app.flags.DEFINE_string('device','/gpu:0',
                          """Device for graph placement""")

tf.app.flags.DEFINE_string('test_path','../data/',
                           """Base directory for test/validation data""")
tf.app.flags.DEFINE_string('filename_pattern_test','val/words-*',
                           """File pattern for test input data""")
tf.app.flags.DEFINE_integer('num_input_threads_eval',4,
                          """Number of readers for input data""")


#Flags used for evalution
tf.app.flags.DEFINE_bool('verbose',True,
                         """Print all predictions, ground truth, and filenames""")
