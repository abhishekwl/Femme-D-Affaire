const { Layer,Network } = require('synaptic');
const fs = require('fs');

const inputLayer = new Layer(2);
const hiddenLayer = new Layer(3);
const outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const newNetwork = new Network({
    input: inputLayer,
    hidden: [ hiddenLayer ],
    outputLayer: outputLayer
});

const LR = 0.3;

fs.readFile('training_data.npy', {encoding: 'utf-8'}, (err,data)=>{
    if(!err) {
        console.log(data);

        lines = data.split('\n');
        lines.map(line=>{
            slopes = line.split(',');
            m1 = slopes[0];
            m2 = slopes[1];
            key = slopes[2];

            newNetwork.activate([ m1, m2 ]);
            newNetwork.propagate(LR, [key]);

        });

    } else console.log(err);
});