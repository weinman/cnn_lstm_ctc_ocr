# Frequently Asked Questions

## Test results aren't good. What's wrong?

The default training schedule in the provided `Makefile` is likely to get caught in a bad local minimum. See [this comment](https://github.com/weinman/cnn_lstm_ctc_ocr/issues/42#issuecomment-428791521) and [this example result](https://github.com/weinman/cnn_lstm_ctc_ocr/issues/42#issuecomment-428793851) in the issues for ideas on how to solve this problem.

## Is there a pretrained model available?

No. I train using a platform with multiple GPUS, and the graph in the training checkpoints reflects this; they are not suitable for general use. Hence, if someone wrote script that harnesses only the model parameters necessary for test, I would gladly use it to package a test model. I don't have time nor need for such a script myself.

## How can you do fine tuning?

After pre-training, use the `--tune_from` flag to specify the source model and the `--tune_scope` flag to restrict the sets of parameters used for tuning.

The following is a basic example:

```sh
# Pre-train the initial model
python train.py --output=../data/pretrained # other flags as desired
# Tune only LSTMs
python train.py --output=../data/finetuned --tune_from=../data/pretrained/model.ckpt-TTTTT --tune_scope=rnn # other flags as desired
```

Note that `TTTTT` should be replaced with the step/checkpoint you want to fine tune from.

See the code in [`model.py`](src/model.py) for uses of `tf.variable_scope` for possible restrictions (or how you might add your own).

