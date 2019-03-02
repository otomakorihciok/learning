import TensorFlow

let hiddenSize = 10

struct Model: Layer {
  var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
  var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
  var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: relu)

  @differentiable
  func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    let l1 = layer1.applied(to: input, in: context)
    let l2 = layer2.applied(to: l1, in: context)
    return layer3.applied(to: l2, in: context)
  }
}

let optimizer = SGD<Model, Float>(learningRate: 0.02)
var classifier = Model()
let context = Context(learningPhase: .training)

let x: Tensor<Float> = Tensor<Float>([[0.1, 0.2, 0.1, 0.3], [1.3, 2.1, 0.5, 3.9]])
let y: Tensor<Float> = Tensor<Float>([[1, 2, 1], [3, 1, 5]])

for _ in 0..<1000 {
  let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
    let ŷ = classifier.applied(to: x, in: context)
    let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
    print("Loss: \(loss)")
    return loss
  }

  optimizer.update(&classifier.allDifferentiableVariables, along: 𝛁model)
}
