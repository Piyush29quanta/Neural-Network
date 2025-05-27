const dataset = [
    {inputs:[0,0],target:0},
    {inputs:[0,1],target:1},
    {inputs:[1,0],target:1},
    {inputs:[1,1],target:0},
];

function sigmoid(x){
    return 1/(1+Math.exp(-x))
}

function dsigmoid(y){
    return y*(1-y);
}

let weights_ih = [
    [Math.random()*2-1,Math.random()*2-1],
    [Math.random()*2-1,Math.random()*2-1]
]

let bias_h = [Math.random()*2-1,Math.random()*2-1];

let weights_ho = [Math.random()*2-1,Math.random()*2-1];

let bias_o = Math.random()*2-1;

function feedForward(inputs){

    let hidden = [];
    for(let i=0; i<2;i++){
        let sum = 0;
        for(let j=0;j<2;j++){
            sum += inputs[j]*weights_ih[j][i]; 
        }
        sum += bias_h[i];
        hidden[i] = sigmoid(sum);
    }

    let output = 0;
    for(let i=0;i<2;i++){
        output += hidden[i]*weights_ho[i]; 
    }
    output += bias_o;
    return  sigmoid(output)
}

console.log("Before training");
for(let data of dataset){
    let output = feedForward(data.inputs);
    console.log(`Input:${data.inputs}-> Output:${output.toFixed(4)}`);
}

function train(inputs,target, learnigRate = 0.1){
    let hidden = [];
    for(let i=0; i<2;i++){
        let sum =0;
        for(let j=0;j<2;j++){
            sum += inputs[j]*weights_ih[j][i];
        }
        sum +=bias_h[i];
        hidden[i] = sigmoid(sum);
    }

    let outputSum = 0;
    for(let i=0;i<2;i++){
        outputSum += hidden[i]*weights_ho[i];
    }
    outputSum += bias_o;
    let output = sigmoid(outputSum);

    let error = target- output;
    let d_output = error*dsigmoid(output);

    for(let i=0;i<2; i++){
        weights_ho[i] += hidden[i]*d_output*learnigRate;
    }
    bias_o += d_output*learnigRate;

    let d_hidden = [];
    for (let i=0;i<2;i++){
        let hidden_error = weights_ho[i]*d_output;
        d_hidden[i] = hidden_error*dsigmoid(hidden[i]);
    }
    for(let i=0;i<2;i++){
        for(let j=0;j<2;j++){
            weights_ih[j][i] += inputs[j]*d_hidden[i]*learnigRate;
        }
        bias_h[i] += d_hidden[i]*learnigRate;
    }
}

for(let epoch =0;epoch<10000;epoch++){
    for(let data of dataset){
        train(data.inputs,data.target);
    }
}

console.log("\nAfter training");
for(let data of dataset){
    let output = feedForward(data.inputs);
    console.log(`Input:${data.inputs}-> Output:${output.toFixed(4)}`);
}