package com.dof.nn.loader.image;

import com.dof.nn.core.NeuralNetwork;
import com.dof.nn.loader.BatchData;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import javax.imageio.ImageIO;


public class ImageWriter {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("usage: [app] <MNIST DATA DIRECTORY>");
            return;
        }

        String directory = args[0];
        File dir = new File(directory);

        if (!new File(args[0]).isDirectory()) {
            try {
                System.out.println(dir.getCanonicalPath() + " is not a directory.");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }

            return;
        }

        new ImageWriter().run(directory);
    }

    private int convertOneHotToInt(double[] labelData, int offset, int oneHotSize) {

        double maxValue = 0;
        int maxIndex = 0;

        for (int i = 0; i < oneHotSize; i++) {
            if (labelData[offset + i] > maxValue) {
                maxValue = labelData[offset + i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public void run(String directory) {
        final Path trainImages = Path.of(directory, "train-images.idx3-ubyte");
        final Path trainLabels = Path.of(directory, "train-labels.idx1-ubyte");
        final Path testImages = Path.of(directory, "t10k-images.idx3-ubyte");
        final Path testLabels = Path.of(directory, "t10k-labels.idx1-ubyte");

        int batchSize = 900;

        ImageLoader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
        ImageLoader testLoader = new ImageLoader(testImages, testLabels, batchSize);

        ImageLoader loader = testLoader;

        ImageMetaData metaData = loader.open();

        var neuralNetwork = NeuralNetwork.load("mnistNeural0.net");

        int imageWidth = metaData.getWidth();
        int imageHeight = metaData.getHeight();

        int labelSize = metaData.getExpectedSize();

        for (int i = 0; i < metaData.getNumberBatches(); i++) {
            BatchData batchData = testLoader.readBatch();

            var numberImages = metaData.getItemsRead();

            int horizontalImages = (int) Math.sqrt(numberImages);

            while (numberImages % horizontalImages != 0) {
                ++horizontalImages;
            }

            int verticalImages = numberImages / horizontalImages;

            int canvasWidth = horizontalImages * imageWidth;
            int canvasHeight = verticalImages * imageHeight;

            String montagePath = String.format("montage%d.jpg", i);
            System.out.println("Writing " + montagePath);

            var montage = new BufferedImage(canvasWidth, canvasHeight, BufferedImage.TYPE_INT_RGB);

            double[] pixelData = batchData.getInputBatch();
            double[] labelData = batchData.getExpectedBatch();

            int imageSize = imageWidth * imageHeight;

            boolean[] correct = new boolean[numberImages];

            for (int n = 0; n < numberImages; n++) {
                double[] singleImage = Arrays.copyOfRange(pixelData, n * imageSize, (n + 1) * imageSize);
                double[] singleLabel = Arrays.copyOfRange(labelData, n * labelSize, (n + 1) * labelSize);

                double[] predictedLabel = neuralNetwork.predict(singleImage);

                int predicted = convertOneHotToInt(predictedLabel, 0, labelSize);
                int actual = convertOneHotToInt(singleLabel, 0, labelSize);

                correct[n] = predicted == actual;
            }

            for (int pixelIndex = 0; pixelIndex < pixelData.length; pixelIndex++) {
                int imageNumber = pixelIndex / imageSize;
                int pixelNumber = pixelIndex % imageSize;

                int montageRow = imageNumber / horizontalImages;
                int montageCol = imageNumber % horizontalImages;

                int pixelRow = pixelNumber / imageWidth;
                int pixelCol = pixelNumber % imageWidth;

                int x = montageCol * imageWidth + pixelCol;
                int y = montageRow * imageHeight + pixelRow;

                double pixelValue = pixelData[pixelIndex];
                int color = (int) (0x100 * pixelValue);
                int pixelColor = 0;

                if (correct[imageNumber]) {
                    pixelColor = color;
                } else {
                    pixelColor = (color << 16); //+ (color << 8) + color;
                }

                montage.setRGB(x, y, pixelColor);
            }

            try {
                ImageIO.write(montage, "jpg", new File(montagePath));
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }

            StringBuilder sb = new StringBuilder();

            for (int labelIndex = 0; labelIndex < numberImages; labelIndex++) {

                if (labelIndex % horizontalImages == 0) {
                    sb.append("\n");
                }

                int label = convertOneHotToInt(labelData, labelIndex * labelSize, labelSize);
                sb.append(String.format("%d ", label));
            }

            String labelPath = String.format("labels%d.txt", i);
            System.out.println("Writing " + labelPath);

            try {
                FileWriter fw = new FileWriter(labelPath);
                fw.write(sb.toString());
                fw.close();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        loader.close();
    }
}