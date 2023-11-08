package com.dof.nn;

import com.dof.nn.loader.Loader;
import com.dof.nn.loader.image.ImageLoader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class App {

    public static void main(String[] args) throws IOException {
        if (args.length == 0 || !new File(args[0]).isDirectory()) {
            System.out.println("usage: [app] <MNIST DATA DIRECTORY>");
            return;
        }
        System.out.println(new File(args[0]).getCanonicalPath());
        System.out.println("Hello");

        String directory = args[0];

        final Path trainImages = Path.of(directory, "train-images.idx3-ubyte");
        final Path trainLabels = Path.of(directory, "train-labels.idx1-ubyte");
        final Path testImages = Path.of(directory, "t10k-images.idx3-ubyte");
        final Path testLabels = Path.of(directory, "t10k-labels.idx1-ubyte");

        Loader trainLoader = new ImageLoader(trainImages, trainLabels, 32);
        Loader testLoader = new ImageLoader(testImages, testLabels, 32);

        trainLoader.open();
        testLoader.open();

        trainLoader.close();
        testLoader.close();
    }
}
