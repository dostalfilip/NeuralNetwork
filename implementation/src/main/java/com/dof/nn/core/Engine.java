package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import java.util.LinkedList;
import java.util.Random;

public class Engine {
    private LinkedList<Transform> transforms = new LinkedList<>();
    private LinkedList<Matrix> weights = new LinkedList<>();
    private LinkedList<Matrix> biases = new LinkedList<>();

    private LossFunction lossFunction = LossFunction.CROSSENTROPY;
    private boolean storeInputError = false;

    BatchResult runForwards(Matrix input) {

        BatchResult batch = new BatchResult();
        Matrix output = input;
        int denseIndex = 0;

        batch.addIo(output);

        for (var t : transforms) {
            if (t == Transform.DENSE) {
                Matrix weight = weights.get(denseIndex);
                Matrix bias = biases.get(denseIndex);
                output = weight.multiply(output).modify((row, col, value) -> value + bias.get(row));
                ++denseIndex;
            } else if (t == Transform.RELU) {
                output = output.modify(value -> value > 0 ? value : 0);
            } else if (t == Transform.SOFTMAX) {
                output = output.softMax();
            }

            batch.addIo(output);
        }
        return batch;
    }

    public void runBackwards(BatchResult batchResult, Matrix expected) {

        var transformsIt = transforms.descendingIterator();

        if (lossFunction != LossFunction.CROSSENTROPY || transforms.getLast() != Transform.SOFTMAX) {
            throw new UnsupportedOperationException("Loss function must be cross entropy and last transform must be softmax");
        }

        var ioIt = batchResult.getIo().descendingIterator();
        var weightsIt = weights.descendingIterator();
        Matrix softMaxOutput = ioIt.next();
        Matrix error = softMaxOutput.apply((index, value) -> value - expected.get(index));

        while (transformsIt.hasNext()) {
            Transform transform = transformsIt.next();

            Matrix input = ioIt.next();

            switch (transform) {
                case DENSE -> {
                    Matrix weight = weightsIt.next();
                    if (weightsIt.hasNext() || storeInputError) {
                        error = weight.transpose().multiply(error);
                    }
                }
                case RELU -> {
                    error = error.apply((index, value) -> input.get(index) > 0 ? value : 0);
                }
                case SOFTMAX -> {
                }
                default -> throw new UnsupportedOperationException("Not implemented");
            }

//            System.out.println(transform);
        }

        if (storeInputError) {
            batchResult.setInputError(error);
        }
    }

    public void add(Transform transform, double... params) {
        Random random = new Random();

        if (transform == Transform.DENSE) {
            int numberNeurons = (int) params[0];
            int weightsPerNeuron = weights.isEmpty() ? (int) params[1] : weights.getLast().getRows();

            Matrix weight = new Matrix(numberNeurons, weightsPerNeuron, i -> random.nextGaussian());
            Matrix bias = new Matrix(numberNeurons, weightsPerNeuron, i -> random.nextGaussian());

            weights.add(weight);
            biases.add(bias);
        }
        transforms.add(transform);
    }

    public boolean isStoreInputError() {
        return storeInputError;
    }

    public void setStoreInputError(boolean storeInputError) {
        this.storeInputError = storeInputError;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();

        int weightIndex = 0;
        for (var t : transforms) {
            sb.append(t);
            if (t == Transform.DENSE) {
                sb.append(" ").append(weights.get(weightIndex).toString(false));
                weightIndex++;
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
