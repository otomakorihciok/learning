# Cifar-10

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory focuses on how to use TensorFlow Estimators to train and
evaluate a CIFAR-10 ResNet model on:

- A single host with one CPU;
- A single host with multiple GPUs;
- Multiple hosts with CPU or multiple GPUs;

Before trying to run the model we highly encourage you to read all the README.

## Prerequisite

1. [Install](https://www.tensorflow.org/install/) TensorFlow version 1.9.0 or
   later.

2. Download the CIFAR-10 dataset and generate TFRecord files using the provided
   script. The script and associated command below will download the CIFAR-10
   dataset and then generate a TFRecord for the training, validation, and
   evaluation datasets.

```shell
python generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data
```

After running the command above, you should see the following files in the
--data-dir (`ls -R cifar-10-data`):

- train.tfrecords
- validation.tfrecords
- eval.tfrecords

## Training on a single machine with GPUs or CPU

Run the training on CPU only. After training, it runs the evaluation.

```bash
python -m trainer.cifar10_main.py --data-dir=${PWD}/cifar-10-data \
                       --job-dir=/tmp/cifar10 \
                       --train-steps=1000
```

There are more command line flags to play with; run
`python cifar10_main.py --help` for details.

## Running on Google Cloud Machine Learning Engine

This example can be run on Google Cloud Machine Learning Engine (ML Engine),
which will configure the environment and take care of running workers,
parameters servers, and masters in a fault tolerant way.

To install the command line tool, and set up a project and billing, see the
quickstart [here](https://cloud.google.com/ml-engine/docs/quickstarts/command-line).

You'll also need a Google Cloud Storage bucket for the data. If you followed the
instructions above, you can just run:

```bash
MY_BUCKET=gs://<my-bucket-name>
gsutil cp -r ${PWD}/cifar-10-data $MY_BUCKET/
```

Then run the following command from the `tutorials/image` directory of this
repository (the parent directory of this README):

### Run on single GPU machine

```bash
gcloud ml-engine jobs submit training <job-name> \
    --runtime-version 1.12 \
    --python-version 3.5 \
    --job-dir=$MY_BUCKET/model_dirs/<job-name> \
    --config config.yaml \
    --package-path trainer/ \
    --module-name trainer.cifar10_main \
    -- \
    --data-dir=$MY_BUCKET/cifar-10-data \
    --num-gpus=1 \
    --train-steps=1000
```

### Run on multi GPU machine

```bash
gcloud ml-engine jobs submit training <job-name> \
    --runtime-version 1.12 \
    --python-version 3.5 \
    --job-dir=$MY_BUCKET/model_dirs/<job-name> \
    --config dist-config.yaml \
    --package-path trainer/ \
    --module-name trainer.cifar10_main \
    -- \
    --data-dir=$MY_BUCKET/cifar-10-data \
    --num-gpus=4 \
    --train-steps=1000
```

## Deploy model on Google Cloud Machine Learning Engine

```bash
gcloud ml-engine models create <model-name>
gcloud ml-engine versions create <version-name> \
    --model <model-name> --origin <model-path>
```

## Test deployed model

```bash
gcloud ml-engine predict --model <model-name> \
--version <version-name> \
--json-instances <json-file-path>
```
