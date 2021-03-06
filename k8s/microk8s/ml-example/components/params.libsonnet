{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    "chainer-operator": {
      createRbac: 'true',
      image: 'gcr.io/kubeflow-images-public/chainer-operator:v0.3.0',
      name: 'chainer-operator',
      serviceAccountName: 'null',
      stderrthreshold: 'INFO',
      v: 2,
    },
    "chainer-job-simple": {
      args: '/train_mnist.py,-e,2,-b,1000,-u,100',
      backend: 'mpi',
      command: 'python3',
      gpus: 0,
      image: 'null',
      name: 'chainer-job-simple',
      workerSetName: 'ws',
      workers: 0,
    },
    "tf-job-operator": {
      cloud: "null",
      deploymentNamespace: "ml-dev",
      deploymentScope: "namescope",
      name: "tf-job-example",
      tfDefaultImage: "null",
      tfJobImage: "gcr.io/kubeflow-images-public/tf_operator:kubeflow-tf-operator-postsubmit-v1beta1-c284947-309-b42b",
      tfJobUiServiceType: "ClusterIP",
    },
    "tf-job-simple": {
      name: "tf-job-simple-cnn-1",
    },
  },
}
