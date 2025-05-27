// Sigmoid activation function and its derivative
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  return x * (1 - x);
}

// Training data (OR logic)
const inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];

const outputs = [0, 1, 1, 1];

// Initialize weights and bias randomly
let weights = [Math.random(), Math.random()];
let bias = Math.random();

// Hyperparameters
const learningRate = 0.1;
const epochs = 10000;

// Training loop
for (let epoch = 0; epoch < epochs; epoch++) {
  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];
    const target = outputs[i];

    // Forward pass
    const weightedSum = input[0] * weights[0] + input[1] * weights[1] + bias;
    const output = sigmoid(weightedSum);

    // Backpropagation
    const error = target - output;
    const delta = error * sigmoidDerivative(output);

    // Update weights and bias
    weights[0] += learningRate * delta * input[0];
    weights[1] += learningRate * delta * input[1];
    bias += learningRate * delta;
  }

  // Print loss occasionally
  if (epoch % 1000 === 0) {
    let totalLoss = 0;
    for (let i = 0; i < inputs.length; i++) {
      const output = sigmoid(inputs[i][0] * weights[0] + inputs[i][1] * weights[1] + bias);
      totalLoss += Math.pow(outputs[i] - output, 2);
    }
    console.log(`Epoch ${epoch} - Loss: ${totalLoss.toFixed(4)}`);
  }
}

// Test the model
console.log("\nTesting OR gate:");
inputs.forEach((input, i) => {
  const output = sigmoid(input[0] * weights[0] + input[1] * weights[1] + bias);
  console.log(`${input} => ${Math.round(output)} (raw: ${output.toFixed(4)})`);
});
