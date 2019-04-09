# BioNER

### Quick Start

```
pip install tensor2tensor && t2t-trainer \
  --generate_data \
  --data_dir=~/t2t_data \
  --output_dir=~/t2t_train/mnist \
  --problem=image_mnist \
  --model=shake_shake \
  --hparams_set=shake_shake_quick \
  --train_steps=1000 \
  --eval_steps=100
```

## Suggested Datasets and Models

Below we list a number of tasks that can be solved with T2T when
you train the appropriate model on the appropriate problem.
We give the problem and model below and we suggest a setting of
hyperparameters that we know works well in our setup. We usually
run either on Cloud TPUs or on 8-GPU machines; you might need
to modify the hyperparameters if you run on a different setup.

### Mathematical Language Understanding

For evaluating mathematical expressions at the character level involving addition, subtraction and multiplication of both positive and negative decimal numbers with variable digits assigned to symbolic variables, use

* the [MLU](https://art.wangperawong.com/mathematical_language_understanding_train.tar.gz) data-set:
 `--problem=algorithmic_math_two_variables`

You can try solving the problem with different transformer models and hyperparameters as described in the [paper](https://arxiv.org/abs/1812.02825):
* Standard transformer:
`--model=transformer`
`--hparams_set=transformer_tiny`
* Universal transformer:
`--model=universal_transformer`
`--hparams_set=universal_transformer_tiny`
* Adaptive universal transformer:
`--model=universal_transformer`
`--hparams_set=adaptive_universal_transformer_tiny`


**Problems** consist of features such as inputs and targets, and metadata such
as each feature's modality (e.g. symbol, image, audio) and vocabularies. Problem
features are given by a dataset, which is stored as a `TFRecord` file with
`tensorflow.Example` protocol buffers. All
problems are imported in
[`all_problems.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/all_problems.py)
or are registered with `@registry.register_problem`. Run
[`t2t-datagen`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-datagen)
to see the list of available problems and download them.

### Models

**`T2TModel`s** define the core tensor-to-tensor computation. They apply a
default transformation to each input and output so that models may deal with
modality-independent tensors (e.g. embeddings at the input; and a linear
transform at the output to produce logits for a softmax over classes). All
models are imported in the
[`models` subpackage](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/__init__.py),
inherit from [`T2TModel`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/t2t_model.py),
and are registered with
[`@registry.register_model`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py).

