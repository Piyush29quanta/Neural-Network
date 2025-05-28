// XOR dataset: Each object has two inputs and the expected XOR output (target)
const dataset = [
    {inputs:[0,0],target:0},
    {inputs:[0,1],target:1},
    {inputs:[1,0],target:1},
    {inputs:[1,1],target:0},
];

// Sigmoid activation function: squashes input to range (0, 1)
function sigmoid(x){
    return 1/(1+Math.exp(-x))
}

// Derivative of sigmoid: used for backpropagation
function dsigmoid(y){
    return y*(1-y);
}

// Weights from input layer to hidden layer (2x2 matrix, random values)
let weights_ih = [
    [Math.random()*2-1,Math.random()*2-1],
    [Math.random()*2-1,Math.random()*2-1]
]

// Biases for hidden layer neurons (2 values, random)
let bias_h = [Math.random()*2-1,Math.random()*2-1];

// Weights from hidden layer to output neuron (2 values, random)
let weights_ho = [Math.random()*2-1,Math.random()*2-1];

// Bias for output neuron (single value, random)
let bias_o = Math.random()*2-1;

// Feedforward: calculates output for given inputs
function feedForward(inputs){

    let hidden = [];
    // Calculate hidden layer activations
    for(let i=0; i<2;i++){
        let sum = 0;
        for(let j=0;j<2;j++){
            sum += inputs[j]*weights_ih[j][i]; // Weighted sum from inputs
        }
        sum += bias_h[i]; // Add bias
        hidden[i] = sigmoid(sum); // Activation
    }

    let output = 0;
    // Calculate output neuron activation
    for(let i=0;i<2;i++){
        output += hidden[i]*weights_ho[i]; // Weighted sum from hidden layer
    }
    output += bias_o; // Add bias
    return  sigmoid(output); // Activation
}

// Show outputs before training
console.log("Before training");
for(let data of dataset){
    let output = feedForward(data.inputs);
    console.log(`Input:${data.inputs}-> Output:${output.toFixed(4)}`);
}

// Train the network using backpropagation
function train(inputs,target, learnigRate = 0.1){
    let hidden = [];
    // Feedforward for hidden layer
    for(let i=0; i<2;i++){
        let sum =0;
        for(let j=0;j<2;j++){
            sum += inputs[j]*weights_ih[j][i];
        }
        sum +=bias_h[i];
        hidden[i] = sigmoid(sum);
    }

    // Feedforward for output neuron
    let outputSum = 0;
    for(let i=0;i<2;i++){
        outputSum += hidden[i]*weights_ho[i];
    }
    outputSum += bias_o;
    let output = sigmoid(outputSum);

    // Calculate error and gradient for output neuron
    let error = target- output;
    let d_output = error*dsigmoid(output);

    // Update weights and bias for output neuron
    for(let i=0;i<2; i++){
        weights_ho[i] += hidden[i]*d_output*learnigRate;
    }
    bias_o += d_output*learnigRate;

    // Backpropagate error to hidden layer
    let d_hidden = [];
    for (let i=0;i<2;i++){
        let hidden_error = weights_ho[i]*d_output;
        d_hidden[i] = hidden_error*dsigmoid(hidden[i]);
    }
    // Update weights and biases for hidden layer
    for(let i=0;i<2;i++){
        for(let j=0;j<2;j++){
            weights_ih[j][i] += inputs[j]*d_hidden[i]*learnigRate;
        }
        bias_h[i] += d_hidden[i]*learnigRate;
    }
}

// Train for 10,000 epochs over the dataset
for(let epoch =0;epoch<10000;epoch++){
    for(let data of dataset){
        train(data.inputs,data.target);
    }
}

// Show outputs after training
console.log("\nAfter training");
for(let data of dataset){
    let output = feedForward(data.inputs);
    console.log(`Input:${data.inputs}-> Output:${output.toFixed(4)}`);
}