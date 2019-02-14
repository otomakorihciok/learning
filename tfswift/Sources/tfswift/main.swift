import TensorFlow

struct Model: Layer {
  var l1, l2: Dense<Float>

  init(hiddenSize: Int) {
    l1 = Dense(inputSize: 2, outputSize: hiddenSize, activation: relu)
    l2 = Dense(inputSize: hiddenSize, outputSize: 1, activation: relu)
  }

  @differentiable(wrt: (self, input))
  func applied(to input: Tensor<Float>) -> Tensor<Float> {
    let h1 = l1.applied(to: input)
    return l2.applied(to: h1)
  }
}

let optimizer = SGD<Model, Float>(learningRate: 0.02)
var classifier = Model(hiddenSize: 4)
let x: Tensor<Float> = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
let y: Tensor<Float> = Tensor([0.1, 0.2, 0.3, 0.4])

for _ in 0 ..< 1000 {
  let model = classifier.gradient { classifier -> Tensor<Float> in
    let y_ = classifier.applied(to: x)
    let loss = meanSquaredError(predicted: y_, expected: y)
    print("Loss: \(loss)")
    return loss
  }
  optimizer.update(&classifier, along: model)
}
