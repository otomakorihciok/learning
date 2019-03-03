import TensorFlow

let batchSize: Int32 = 4

let (trainingDataset, testDataset) = loadCIFAR10()
let trainingBatches = trainingDataset.batched(Int64(batchSize))
let testBatches = testDataset.batched(Int64(batchSize))

// Initialize model at first.
var model = PyTorchModel()
let optimizer = SGD<PyTorchModel, Float>(learningRate: 0.001, momentum: 0.9)
let trainingContext = Context(learningPhase: .training)
let inferenceContext = Context(learningPhase: .inference)

for epoch in 1 ... 10 {
  print("Epoch \(epoch), training...")
  var trainingLossSum: Float = 0
  var trainingBatchCount = 0
  for batch in trainingBatches {
    let 𝛁model = model.gradient { model -> Tensor<Float> in
      let ŷ = model.applied(to: batch.second, in: trainingContext)
      let oneHotLabels = Tensor<Float>(
        oneHotAtIndices: batch.first, depth: ŷ.shape[1]
      )
      let loss = softmaxCrossEntropy(logits: ŷ, labels: oneHotLabels)
      trainingLossSum += loss.scalarized()
      trainingBatchCount += 1
      return loss
    }
    optimizer.update(&model.allDifferentiableVariables, along: 𝛁model)
  }

  print("   average loss: \(trainingLossSum / Float(trainingBatchCount))")
  print("Epoch \(epoch), evaluating on test set...")
  var testLossSum: Float = 0
  var testBatchCount = 0
  for batch in testBatches {
    let ŷ = model.applied(to: batch.second, in: inferenceContext)
    let oneHotLabels = Tensor<Float>(
      oneHotAtIndices: batch.first, depth: ŷ.shape[1]
    )
    testLossSum += softmaxCrossEntropy(logits: ŷ, labels: oneHotLabels).scalarized()
    testBatchCount += 1
  }
  print("   average loss: \(testLossSum / Float(testBatchCount))")
}
