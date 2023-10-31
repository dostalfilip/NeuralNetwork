package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralNetworkTest {
    private Random random = new Random();

    @Test
    void testTrainEngine() {
        int inputRows = 500;
        int cols = 32;
        int outputRows = 3;

        Engine engine = new Engine();
        engine.add(Transform.DENSE, 6, inputRows);
        engine.add(Transform.RELU);
        engine.add(Transform.DENSE, outputRows);
        engine.add(Transform.SOFTMAX);

        for (int i = 0; i < 2000; i++) {
            var tm = Util.generateTrainingMatrices(inputRows, outputRows, cols);
            var input = tm.getInput();
            var expected = tm.getOutput();

            BatchResult batchResult = engine.runForwards(input);
            engine.runBackwards(batchResult, expected);
            engine.adjust(batchResult, 0.01);
            engine.evaluate(batchResult, expected);

            double loss = batchResult.getLoss();
            double percentCorrect = batchResult.getPercentCorrect();

            System.out.printf("Loss: %.3f, %% correct:  %.2f\n", loss, percentCorrect);
        }
    }

    interface NeuralNet {
        Matrix apply(Matrix m);
    }

    @Test
    void testWeightGradient() {
        int inputRows = 4;
        int outputRows = 5;

        Matrix weight = new Matrix(outputRows, inputRows, i -> random.nextGaussian());
        Matrix input = Util.generateInputMatrix(inputRows, 1);
        Matrix expected = Util.generateExpectedMatrix(outputRows, 1);

        Matrix output = weight.multiply(input).softMax();

        Matrix calculatedError = output.apply((index, value) -> value - expected.get(index));

        Matrix calculatedWeightGradients = calculatedError.multiply(input.transpose());

        Matrix approximatedWeightGradients = Approximator.weightGradient(
            weight,
            w -> {
                Matrix out = w.multiply(input).softMax();
                return LossFunctions.crossEntropy(expected, out);
            });

        calculatedWeightGradients.setTolerance(0.01);

        assertEquals(calculatedWeightGradients, approximatedWeightGradients);
    }

    @Test
    void testEngine() {

        int inputRows = 5;
        int cols = 6;
        int outputRows = 4;

        Engine engine = new Engine();

        engine.add(Transform.DENSE, 8, 5);
        engine.add(Transform.RELU);
        engine.add(Transform.DENSE, 5);
        engine.add(Transform.RELU);
        engine.add(Transform.DENSE, 4);

        engine.add(Transform.SOFTMAX);
        engine.setStoreInputError(true);

        Matrix input = Util.generateInputMatrix(inputRows, cols);
        Matrix expected = Util.generateExpectedMatrix(outputRows, cols);

        Matrix approximatedError = Approximator.gradient(input, in -> {
            BatchResult batchResult = engine.runForwards(in);
            return LossFunctions.crossEntropy(expected, batchResult.getOutput());
        });

        BatchResult batchResult = engine.runForwards(input);
        engine.runBackwards(batchResult, expected);

        Matrix calculatedError = batchResult.getInputError();
        calculatedError.setTolerance(0.0001);

        System.out.println(calculatedError);
        System.out.println(approximatedError);

        assertEquals(calculatedError, approximatedError);
    }

    @Test
    void testBackPropagation() {
        final int inputRows = 4;
        final int cols = 5;
        final int outputRows = 4;

        Matrix input = new Matrix(inputRows, cols, index -> (random.nextGaussian()));

        Matrix expected = new Matrix(outputRows, cols, index -> 0);

        Matrix weights = new Matrix(outputRows, inputRows, index -> (random.nextGaussian()));
        Matrix biases = new Matrix(outputRows, 1, index -> (random.nextGaussian()));

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(outputRows);

            expected.set(randomRow, col, 1);
        }

        NeuralNet neuralNet = m -> {
            Matrix out = m.apply((index, value) -> value > 0 ? value : 0);
            out = weights.multiply(out); // weights
            out.modify((row, col, value) -> value + biases.get(row)); // biases
            out = out.softMax(); // Softmax activation function
            return out;
        };
        Matrix softMaxOutput = neuralNet.apply(input);

        Matrix approximatedResult = Approximator.gradient(input, in -> {
            Matrix out = neuralNet.apply(in);
            return LossFunctions.crossEntropy(expected, out);
        });

        Matrix calculatedResult = softMaxOutput.apply((index, value) -> value - expected.get(index));
        calculatedResult = weights.transpose().multiply(calculatedResult);
        calculatedResult = calculatedResult.apply((index, value) -> input.get(index) > 0 ? value : 0);

        assertEquals(approximatedResult, calculatedResult);
    }

    @Test
    void testSoftMaxCrossEntropyGradient() {
        final int rows = 4;
        final int cols = 5;

        Matrix input = new Matrix(rows, cols, index -> (random.nextGaussian()));

        Matrix expected = new Matrix(rows, cols, index -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(rows);

            expected.set(randomRow, col, 1);
        }

        Matrix softMaxOutput = input.softMax();

        Matrix result = Approximator.gradient(input, in -> LossFunctions.crossEntropy(expected, in.softMax()));

        result.forEach((index, value) -> {
            double softMaxOutputValue = softMaxOutput.get(index);
            double expectedValue = expected.get(index);

            assertTrue(Math.abs(value - (softMaxOutputValue - expectedValue)) < 0.01);
            System.out.println(value + ", " + (softMaxOutputValue - expectedValue));
        });

    }

    @Test
    void testApproximator() {
        final int rows = 4;
        final int cols = 5;

        Matrix input = new Matrix(rows, cols, index -> (random.nextGaussian())).softMax();

        Matrix expected = new Matrix(rows, cols, index -> 0);

        for (int col = 0; col < expected.getCols(); col++) {
            int randomRow = random.nextInt(rows);

            expected.set(randomRow, col, 1);
        }

        Matrix result = Approximator.gradient(input, in -> LossFunctions.crossEntropy(expected, in));

        input.forEach((index, value) -> {
            double resultValue = result.get(index);
            double expectedValue = expected.get(index);

            if (expectedValue < 0.001) {
                assertTrue(Math.abs(resultValue) < 0.01);
            } else {
                assertTrue(Math.abs(resultValue + 1.0 / value) < 0.01);
            }
        });

    }

    @Test
    void testCrossEntropy() {
        double[] expectedValues = {1, 0, 0, 0, 0, 1, 0, 1, 0};
        Matrix expected = new Matrix(3, 3, index -> expectedValues[index]);

        Matrix actual = new Matrix(3, 3, index -> index * index * 0.05).softMax();

        Matrix result = LossFunctions.crossEntropy(expected, actual);

        actual.forEach((row, col, index, value) -> {
            double expectedValue = expected.get(index);
            double loss = result.get(col);
            if (expectedValue > 0.9) {
                assertTrue(Math.abs(-Math.log(value) - loss) < 0.001);
            }
        });

    }

    //@Test
    void testTemp() {

        int inputSize = 5;
        int layer1Size = 6;
        int layer2Size = 4;

        Matrix input = new Matrix(inputSize, 1, i -> random.nextGaussian());
        Matrix layer1Weights = new Matrix(layer1Size, input.getRows(), i -> random.nextGaussian());
        Matrix layer1Biases = new Matrix(layer1Size, 1, i -> random.nextGaussian());

        Matrix layer2Weights = new Matrix(layer2Size, layer1Weights.getRows(), i -> random.nextGaussian());
        Matrix layer2Biases = new Matrix(layer2Size, 1, i -> random.nextGaussian());

        var output = input;
        System.out.println(output);

        output = layer1Weights.multiply(output);
        System.out.println(output);

        output = output.modify((row, col, value) -> value + layer1Biases.get(row));
        System.out.println(output);

        output = output.modify((value) -> value > 0 ? value : 0);
        System.out.println(output);

        // layer 2

        output = layer2Weights.multiply(output);
        System.out.println(output);

        output = output.modify((row, col, value) -> value + layer2Biases.get(row));
        System.out.println(output);

        output = output.softMax();
        System.out.println(output);

    }

    @Test
    void testAddBias() {

        Matrix input = new Matrix(3, 3, index -> (index + 1));
        Matrix weights = new Matrix(3, 3, index -> (index + 1));
        Matrix biases = new Matrix(3, 1, index -> (index + 1));

        Matrix result = weights.multiply(input).modify((row, col, value) -> value + biases.get(row));

        double[] expectedValues = {+31.00000, +37.00000, +43.00000, +68.00000, +83.00000, +98.00000, +105.00000, +129.00000, +153.00000};

        Matrix expected = new Matrix(3, 3, index -> (expectedValues[index]));

        assertEquals(result, expected);
    }

    @Test
    void testReLu() {

        final int numberNeurons = 5;
        final int numberInputs = 6;
        final int inputSize = 4;

        Matrix input = new Matrix(inputSize, numberInputs, index -> (random.nextDouble()));
        Matrix weights = new Matrix(numberNeurons, inputSize, index -> (random.nextGaussian()));
        Matrix biases = new Matrix(numberNeurons, 1, index -> (random.nextGaussian()));

        Matrix result1 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row));
        Matrix result2 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row)).modify(value -> value > 0 ? value : 0);

        result2.forEach((index, value) -> {
            double originalValue = result1.get(index);
            if (originalValue > 0) {
                assertTrue(Math.abs(originalValue - value) < 0.000001);
            } else {
                assertTrue(Math.abs(value) < 0.000001);
            }
        });
    }


}
