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
  },
}