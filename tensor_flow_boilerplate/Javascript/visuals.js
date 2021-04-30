export default class Visuals {

    paused = false;
    showHeatmap = false;
    showInitialNetwork = false;
    weights;
    biases;
    oldWeights;
    oldBiases;
    initialWeights;
    initialBiases;
    heatmapWeights;
    heatmapBiases;
    startPos = [100, 400];

    getNeurons(neuralNet) {
        let nodes = [];
        nodes.push(neuralNet.layers[0].input.shape[1]);
        for (let i = 0; i < neuralNet.layers.length; i++) {
            nodes.push(neuralNet.layers[i].units);
        }

        this.neurons = [];

        for (let i = 0; i < nodes.length; i++) {
            let temp = [];
            for (let j = 0; j < nodes[i]; j++) {
                let pos = [];
                let x = this.startPos[0] + i * 100
                let y = this.startPos[1] + j * 50 - (nodes[i] * 50) / 2;
                pos = [x, y]
                temp.push(pos);
            }

            this.neurons.push(temp);
        }
    }

    resetVisuals(neuralNet) {
        this.initializeHeatmap();
        this.updateWeights(neuralNet);
        background(207, 9, 92);
        this.drawConections();
        this.drawNeurons();
        this.initialWeights = this.weights
        this.initialBiases = this.biases
    }

    switchToHeatmap() {
        this.showHeatmap = !this.showHeatmap;
        background(207, 9, 92);
        this.drawConections();
        this.drawNeurons();
        if (this.showHeatmap) {
            document.getElementById("heatmapButton").style.backgroundColor = "#FAA100"
            document.getElementById("heatmapButton").innerHTML = "Show Current Network"
        } else {
            document.getElementById("heatmapButton").style.backgroundColor = "#22E027"
            document.getElementById("heatmapButton").innerHTML = "Show Heatmap"
        }
    }

    switchToInitialNetwork() {
        this.showInitialNetwork = !this.showInitialNetwork
        background(207, 9, 92);
        this.drawConections();
        this.drawNeurons();
        if (this.showInitialNetwork) {
            document.getElementById("initialNetworkButton").style.backgroundColor = "#FAA100"
            document.getElementById("initialNetworkButton").innerHTML = "Show Current Network"
        } else {
            document.getElementById("initialNetworkButton").style.backgroundColor = "#22E027"
            document.getElementById("initialNetworkButton").innerHTML = "Show Initial Network"
        }
    }


    updateWeights(neuralNet) {
        this.oldWeights = this.weights;
        this.oldBiases = this.biases;
        this.weights = [];
        this.biases = [];
        for (let i = 0; i < neuralNet.layers.length; i++) {
            let a = neuralNet.layers[i].getWeights()[0];
            let b = a.arraySync();
            this.weights.push(b);

            let c = neuralNet.layers[i].getWeights()[1];
            let d = c.arraySync();
            this.biases.push(d);
        }
    }

    initializeHeatmap() {
        this.initaializeHeatmapWeights();
        this.initializeHeatmapBiases();
    }

    initializeHeatmapBiases() {
        let temp = []
        for (let i = 0; i < this.biases.length; i++) {
            let temp2 = [];
            for (let j = 0; j < this.biases[i].length; j++) {
                temp2.push(0);
            }
            temp.push(temp2)
        }
        this.heatmapBiases = temp;
    }

    initaializeHeatmapWeights() {
        let temp = [];
        for (let i = 0; i < this.weights.length; i++) {
            let temp3 = [];
            for (let j = 0; j < this.weights[i].length; j++) {
                let temp2 = [];
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    temp2.push(0);
                }
                temp3.push(temp2);
            }
            temp.push(temp3)
        }
        this.heatmapWeights = temp;
    }





    updateHeatmap() {
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    this.heatmapWeights[i][j][k] += Math.abs(this.weights[i][j][k] - this.oldWeights[i][j][k])
                }
            }
        }

        console.log(this.heatmapBiases)

        for (let i = 0; i < this.heatmapBiases.length; i++) {
            for (let j = 0; j < this.heatmapBiases[i].length; j++) {
                this.heatmapBiases[i][j] += Math.abs(this.biases[i][j] - this.oldBiases[i][j]);

            }

        }
    }


    pauseGame() {
        if (this.paused) {
            frameRate(5);
            this.paused = false
            document.getElementById("pauseButton").style.backgroundColor = "#22E027"
            document.getElementById("pauseButton").innerHTML = "running"
        } else {
            frameRate(0);
            this.paused = true
            document.getElementById("pauseButton").style.backgroundColor = "red"
            document.getElementById("pauseButton").innerHTML = "paused"
        }
    }

    drawNeurons() {
        if (this.showHeatmap) {
            this.iterateThroughNeurons(this.biases, this.drawSingleNeuron)
        } else if (this.showInitialNetwork) {
            this.iterateThroughNeurons(this.initialBiases, this.drawSingleNeuron)
        } else {
            this.iterateThroughNeurons(this.biases, this.drawSingleNeuron)
        }
    }


    iterateThroughNeurons(wantedArray, wantedFunction) {
        for (let i = 0; i < this.neurons.length; i++) {
            for (let j = 0; j < this.neurons[i].length; j++) {
                wantedFunction(i, j, wantedArray);
            }
        }
    }


    drawSingleNeuron(i, j, Array) {
        if (i > 0) {
            if (visuals.showHeatmap == true) {
                fill(38.6, 100, 98)
                ellipse(visuals.neurons[i][j][0], visuals.neurons[i][j][1], visuals.heatmapBiases[i - 1][j] * 20)
                return
            } else {
                if (Array[i - 1][j] > 0) {
                    fill(242, 100, map(Array[i - 1][j], 0, 2, 0, 100))
                } else {
                    fill(3, 100, map(Array[i - 1][j], -2, 0, 100, 0))
                }
            }
        } else {
            fill(321, 69, 69)
        }
        noStroke();
        ellipse(visuals.neurons[i][j][0], visuals.neurons[i][j][1], 20)
    }


    drawConections() {
        if (this.showInitialNetwork) {
            this.iterateThroughWeights(this.initialWeights, this.drawSingleWeight)
        } else if (this.showHeatmap) {
            this.iterateThroughWeights(this.heatmapWeights, this.drawSingleHeatmapWeight)
        } else {
            this.iterateThroughWeights(this.weights, this.drawSingleWeight)
        }
    }

    iterateThroughWeights(wantedArray, wantedFunction) {
        for (let i = 0; i < this.neurons.length - 1; i++) {
            for (let j = 0; j < this.neurons[i].length; j++) {
                for (let k = 0; k < this.neurons[i + 1].length; k++) {
                    wantedFunction(i, j, k, wantedArray)
                }
            }
        }
    }

    drawSingleWeight(i, j, k, wantedArray) {
        if (wantedArray[i][j][k] > 0) {
            stroke(242, 100, 100)
        } else {
            stroke(3, 100, 100);
        }
        strokeWeight(Math.abs((wantedArray[i][j][k]) * 2) * slider2.value() / 50);
        line(visuals.neurons[i][j][0], visuals.neurons[i][j][1], visuals.neurons[i + 1][k][0], visuals.neurons[i + 1][k][1]);
    }

    drawSingleHeatmapWeight(i, j, k) {
        stroke(39, 100, 100)
        strokeWeight(Math.abs((visuals.heatmapWeights[i][j][k]) * 2) * slider2.value() / 50);
        line(visuals.neurons[i][j][0], visuals.neurons[i][j][1], visuals.neurons[i + 1][k][0], visuals.neurons[i + 1][k][1]);
    }
}