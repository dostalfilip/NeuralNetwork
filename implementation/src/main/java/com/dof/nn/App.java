package com.dof.nn;

import com.dof.nn.core.NeuralNetwork;
import com.dof.nn.core.Transform;
import com.dof.nn.loader.Loader;
import com.dof.nn.loader.MetaData;
import com.dof.nn.loader.image.ImageLoader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class App {

    public static void main(String[] args) {

        final String filename = "mnistNeural0.net";

        if (args.length == 0) {
            System.out.println("usage: [app] <MNIST DATA DIRECTORY>");
            return;
        }

        String directoryPathString = args[0];
        File dir = new File(directoryPathString);

        if (!dir.isDirectory()) {
            try {
                System.out.println(dir.getCanonicalPath() + " is not a directory.");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }

            return;
        }

        final Path trainImages = Path.of(directoryPathString, "train-images.idx3-ubyte");
        final Path trainLabels = Path.of(directoryPathString, "train-labels.idx1-ubyte");
        final Path testImages = Path.of(directoryPathString, "t10k-images.idx3-ubyte");
        final Path testLabels = Path.of(directoryPathString, "t10k-labels.idx1-ubyte");

        Loader trainLoader = new ImageLoader(trainImages, trainLabels, 32);
        Loader testLoader = new ImageLoader(testImages, testLabels, 32);

        MetaData metaData = trainLoader.open();
        int inputSize = metaData.getInputSize();
        int outputSize = metaData.getExpectedSize();
        trainLoader.close();

        NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);

        if (neuralNetwork == null) {
            System.out.println("Unable to load neural network from saved. Creating from scratch.");

            neuralNetwork = new NeuralNetwork();
            neuralNetwork.setScaleInitialWeights(0.2);
            neuralNetwork.setThreads(4);
            neuralNetwork.setEpochs(100);
            neuralNetwork.setLearningRates(0.02, 0.001);

            neuralNetwork.add(Transform.DENSE, 200, inputSize);
            neuralNetwork.add(Transform.RELU);
            neuralNetwork.add(Transform.DENSE, outputSize);
            neuralNetwork.add(Transform.SOFTMAX);

        } else {
            System.out.println("Loaded from " + filename);
        }

        System.out.println(neuralNetwork);

        neuralNetwork.fit(trainLoader, testLoader);

        if (neuralNetwork.save(filename)) {
            System.out.println("Saved to " + filename);
        } else {
            System.out.println("Unable to save to " + filename);
        }
    }

}