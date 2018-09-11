# History

You may check out the following tags for previous versions of this software.

### `tf-1.10`

Massively overhauled for Tensorflow 1.10.

This version uses the TensorFlow
[Dataset](https://www.tensorflow.org/guide/datasets) for fast
I/O. Training, testing, validation, and prediction use a custom
[Estimator](https://www.tensorflow.org/guide/estimators).

The input pipeline was refactored to account for a dynamically
generated data source
([MapTextSynthesizer](https://github.com/weinman/MapTextSynthesizer)),
with supporting command-line flags were added to `train.py`.

### `tf-1.8`

Updated for Tensorflow 1.8.

This version uses the original TensorFlow
[Reader](https://www.tensorflow.org/versions/r1.8/api_guides/python/io_ops#Readers)
and
[QueueRunner](https://www.tensorflow.org/versions/r1.8/api_guides/python/reading_data#_QueueRunner)
mechanisms for fast, parallel I/O. For training it uses a
straightforward
[MonitoredTrainingSession](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/train/MonitoredTrainingSession). Testing
and evaluation manually manage sessions and checkpoints.

### `tf-1.2`

Updated for Tensorflow 1.1/1.2.

This version uses the original TensorFlow
[Reader](https://www.tensorflow.org/versions/r1.2/api_guides/python/io_ops#Readers)
and
[QueueRunner](https://www.tensorflow.org/versions/r1.2/api_guides/python/reading_data#_QueueRunner)
mechanisms for fast, parallel I/O. For training it uses a
[Supervisor](https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/train/Supervisor)
with a [managed
session](https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/train/Supervisor#managed_session). Testing
and evaluation manually manage sessions and checkpoints.

### `5fa06b7e`

Initial version for Tensorflow 1.1. Lacked `validate.py` and a few
other bug fixes added to tag `tf-1.2`.
