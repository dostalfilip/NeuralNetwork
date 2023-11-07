package com.dof.nn;

import com.dof.nn.core.NeuralNetwork;
import com.dof.nn.core.Transform;
import com.dof.nn.loader.Loader;
import com.dof.nn.loader.test.TestLoader;

public class App {

    public static void main(String[] args) {
        String filename = "neural1.net";

        System.out.println("Count of available processors: " + Runtime.getRuntime().availableProcessors());

        NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);

        if (neuralNetwork == null) {
            System.out.println("Unable to load neural network from saved. Creating from scratch.");
            int inputRows = 10;
            int outputRows = 3;

            neuralNetwork = new NeuralNetwork();
            neuralNetwork.add(Transform.DENSE, 100, inputRows);
            neuralNetwork.add(Transform.RELU);
            neuralNetwork.add(Transform.DENSE, 50, inputRows);
            neuralNetwork.add(Transform.RELU);
            neuralNetwork.add(Transform.DENSE, outputRows);
            neuralNetwork.add(Transform.SOFTMAX);

            neuralNetwork.setThreads(20);
            neuralNetwork.setEpochs(50);
            neuralNetwork.setLearningRates(0.02, 0.001);

        } else {
            System.out.println("Loading neural network from " + filename);
        }

        System.out.println(neuralNetwork);
        Loader trainLoader = new TestLoader(60_000, 32);
        Loader testLoader = new TestLoader(10_000, 32);

        neuralNetwork.fit(trainLoader, testLoader);

        if (neuralNetwork.save("neural1.net")) {
            System.out.println("Saved to " + filename);
        } else {
            System.out.println("Unable to save to " + filename);
        }

    }
}
