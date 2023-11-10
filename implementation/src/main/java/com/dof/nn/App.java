package com.dof.nn;

import com.dof.nn.loader.Loader;
import com.dof.nn.loader.MetaData;
import com.dof.nn.loader.image.ImageLoader;

import java.io.File;
import java.nio.file.Path;

public class App {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("usage: [app] <MNIST DATA DIRECTORY>");
            return;
        }

        final String directoryPathString = args[0];

        if (!new File(directoryPathString).isDirectory()) {
            System.out.println("'" + directoryPathString + "' is not a directory");
            return;
        }

        final Path trainImages = Path.of(directoryPathString, "train-images.idx3-ubyte");
        final Path trainLabels = Path.of(directoryPathString, "train-labels.idx1-ubyte");
        final Path testImages = Path.of(directoryPathString, "t10k-images.idx3-ubyte");
        final Path testLabels = Path.of(directoryPathString, "t10k-labels.idx1-ubyte");

        Loader trainLoader = new ImageLoader(trainImages, trainLabels, 32);
        Loader testLoader = new ImageLoader(testImages, testLabels, 32);

        trainLoader.open();
        MetaData metaData = testLoader.open();

//        for (int i = 0; i < metaData.getNumberBatches(); i++) {
//            BatchData batchData = trainLoader.readBatch();
//            System.out.println(batchData);
//        }

        trainLoader.close();
        testLoader.close();
    }
}
