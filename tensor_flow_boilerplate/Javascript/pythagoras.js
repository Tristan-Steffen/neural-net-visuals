import Visuals from "./visuals.js"

const visuals = new Visuals();

let neuralNet = tf.sequential();
const adamOpt = tf.train.adam(0.02);


var trainingSet;
let validationSet;
let testSetInput;
let testSetExpected;

window.setup = setup;
window.draw = draw;
window.visuals = visuals;
window.neuralNet = neuralNet;


function setup() {
    createCanvas(windowWidth, windowHeight)
    colorMode(HSB);

    document.getElementById("pauseButton").addEventListener("click", function () {
        visuals.pauseGame();
    });
    document.getElementById("resetButton").addEventListener("click", function () {
        createNeuralNet();
        visuals.resetVisuals(neuralNet);
    });
    document.getElementById("heatmapButton").addEventListener("click", function () {
        visuals.switchToHeatmap();
    });
    document.getElementById("initialNetworkButton").addEventListener("click", function () {
        visuals.switchToInitialNetwork();
    });

    const slider = createSlider(1, 20, 1);
    slider.position(520, 0);
    slider.style('width', '300px');

    const slider2 = createSlider(1, 100, 50);
    slider2.position(520, 20);
    slider2.style('width', '300px');

    initializingTrainingSet();
    createNeuralNet();
    


    testSetInput = tf.tensor([
        [5, 5],
        [7, 3],
        [22, 56],
        [1, 12]
    ])

    testSetExpected = tf.tensor([
        [7.071],
        [7.616],
        [60.166],
        [12.042]
    ])



    print(neuralNet)
    visuals.getNeurons(neuralNet);
    visuals.updateWeights(neuralNet);
    visuals.initializeHeatmap();
    visuals.initialWeights = visuals.weights
    visuals.initialBiases = visuals.biases

    window.slider = slider;
    window.slider2 = slider2;
}


async function train() {
    let setup = {
        epochs: 10
    }
    let answer = await neuralNet.fit(trainingSet, validationSet, setup)
    console.log(answer.history.loss[0]);

}

function createNeuralNet() {

    neuralNet = tf.sequential();

    neuralNet.add(tf.layers.dense({
        units: 5,
        inputShape: [2]
    }));

    neuralNet.add(tf.layers.dense({
        units: 6
    }));

    neuralNet.add(tf.layers.dense({
        units: 7
    }));

    neuralNet.add(tf.layers.dense({
        units: 5
    }));
    neuralNet.add(tf.layers.dense({
        units: 1
    }));

    neuralNet.compile({
        optimizer: adamOpt,
        loss: tf.losses.absoluteDifference
    });
}


function draw() {
    background(207, 9, 92);
    let val = window.slider.value();
    stroke(0, 0, 0)
    strokeWeight(1)
    frameRate(val);
    text(frameCount, 30, 30);

    train().then(
        visuals.updateWeights(neuralNet),
        visuals.updateHeatmap(),
        neuralNet.predict(testSetInput).print());

    visuals.drawConections();
    visuals.drawNeurons();

}




function initializingTrainingSet() {
    let training = [];
    let testing = [];

    for (let i = 0; i < 5000; i++) {
        let x = Math.floor(Math.abs(Math.random() * 100));
        let y = Math.floor(Math.abs(Math.random() * 100));
        let z = Math.sqrt(x * x + y * y);

        training.push([x, y]);
        testing.push(z);
    }

    trainingSet = tf.tensor(training);
    validationSet = tf.tensor(testing);
}