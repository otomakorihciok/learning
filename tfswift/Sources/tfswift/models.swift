import TensorFlow

public struct PyTorchModel: Layer {
  var conv1: Conv2D<Float>
  var pool: MaxPool2D<Float>
  var conv2: Conv2D<Float>
  var dense1: Dense<Float>
  var dense2: Dense<Float>
  var dense3: Dense<Float>

  public init() {
    conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 6), padding: .valid)
    pool = MaxPool2D<Float>(
      poolSize: (2, 2), strides: (2, 2), padding: .valid
    )
    conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), padding: .valid)
    dense1 = Dense<Float>(
      inputSize: 16 * 5 * 5, outputSize: 120, activation: relu
    )
    dense2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    dense3 = Dense<Float>(inputSize: 84, outputSize: 10, activation: { $0 })
  }

  @differentiable
  public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    var tmp = input
    tmp = pool.applied(to: relu(conv1.applied(to: tmp, in: context)), in: context)
    tmp = pool.applied(to: relu(conv2.applied(to: tmp, in: context)), in: context)
    let batchSizse = tmp.shape[0]
    tmp = tmp.reshaped(
      toShape: Tensor<Int32>([batchSizse, Int32(16 * 5 * 5)])
    )
    tmp = dense1.applied(to: tmp, in: context)
    tmp = dense2.applied(to: tmp, in: context)
    tmp = dense3.applied(to: tmp, in: context)
    return tmp
  }
}
