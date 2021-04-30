class Visualize {

  paused = false;
  nodes = [];
  neurons = [];
  weights = [];
  heatmap = [];
  startPos = [100, 400];

  getNeurons() {
    this.nodes = [];
    this.nodes.push(model.layers[0].input.shape[1]);
    for (let i = 0; i < model.layers.length; i++) {
      this.nodes.push(model.layers[i].units);
    }

    for (let i = 0; i < this.nodes.length; i++) {
      let temp = [];
      for (let j = 0; j < this.nodes[i]; j++) {
        let pos = [];
        let x = this.startPos[0] + i * 100
        let y = this.startPos[1] + j * 50 - (this.nodes[i] * 50) / 2;
        pos = [x, y]
        temp.push(pos);
      }
      this.neurons.push(temp);
    }
  }

  updateWeights() {
    this.weights = [];
    for (let i = 0; i < model.layers.length; i++) {
      let a = model.layers[i].getWeights()[0];
      let b = a.arraySync();
      this.weights.push(b);
    }
  }

  pauseGame() {
    if (this.paused) {
      frameRate(5);
      this.paused = false
      document.getElementById("myBtn").style.backgroundColor = "#22E027"
    } else {
      frameRate(0);
      this.paused = true
      document.getElementById("myBtn").style.backgroundColor = "red"
    }
  }

  drawConections() {
    for (let i = 0; i < this.neurons.length - 1; i++) {
      for (let j = 0; j < this.neurons[i].length; j++) {
        for (let k = 0; k < this.neurons[i + 1].length; k++) {
          if (this.weights[i][j][k] > 0) {
            stroke(242, 100, 100)
          } else {
            stroke(3, 100, 100);
          }
          strokeWeight(Math.abs(this.weights[i][j][k]) * 2);
          line(this.neurons[i][j][0], this.neurons[i][j][1], this.neurons[i + 1][k][0], this.neurons[i + 1][k][1]);
        }
      }
    }
  }

  drawNeurons() {
    for (let i = 0; i < this.neurons.length; i++) {
      for (let j = 0; j < this.neurons[i].length; j++) {
        ellipse(this.neurons[i][j][0], this.neurons[i][j][1], 20)
      }
    }
  }
}






let validateSet;
let trainingSet;
const model = tf.sequential();
const visuals = new Visualize();

function setup() {

  document.getElementById("myBtn").addEventListener("click", function () {
    visuals.pauseGame();
  });

  colorMode(HSB);

  slider = createSlider(1, 20, 1);
  slider.position(120, 10);
  slider.style('width', '1000px');


  createCanvas(windowWidth, windowHeight);

  model.add(tf.layers.dense({
    units: 4,
    inputShape: [3]
  }));
  model.add(tf.layers.dense({
    units: 6
  }));
  model.add(tf.layers.dense({
    units: 7
  }));
  model.add(tf.layers.dense({
    units: 7
  }));
  model.add(tf.layers.dense({
    units: 6
  }));
  model.add(tf.layers.dense({
    units: 2
  }));

  trainingSet = tf.tensor([
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
  ])

  validateSet = tf.tensor([
    [0, 0],
    [1, 0],
    [1, 0],
    [1, 1]
  ])


  model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: 'meanSquaredError'
  });


  visuals.getNeurons();
  visuals.updateWeights();
}



async function train() {
  let setup = {
    epochs: 3
  }
  let answer = await model.fit(trainingSet, validateSet, setup)
  console.log(answer.history.loss[0]);
  visuals.updateWeights();
}


function draw() {
  background(207, 9, 92);
  let val = slider.value();
  frameRate(val);
  text(frameCount, 30, 30);
  visuals.drawConections();
  visuals.drawNeurons();
  train().then(
    model.predict(trainingSet).print());
}