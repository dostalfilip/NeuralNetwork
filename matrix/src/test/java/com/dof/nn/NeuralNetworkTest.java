package com.dof.nn;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class NeuralNetworkTest {
    private  Random random = new Random();

    @Test
    void testAddBias() {

        Matrix input = new Matrix(3, 3, index -> (index + 1));
        Matrix weights = new Matrix(3, 3, index -> (index + 1));
        Matrix biases = new Matrix(3, 1, index -> (index + 1));

        Matrix result = weights.multiply(input).modify((row, col, value) -> value + biases.get(row));

        double[] expectedValues = {+31.00000, +37.00000, +43.00000, +68.00000, +83.00000, +98.00000, +105.00000, +129.00000, +153.00000};

        Matrix expected = new Matrix(3, 3, index -> (expectedValues[index]));

        assertTrue(result.equals(expected));
    }

    @Test
    void testReLu() {

        final int numberNeurons = 5;
        final int numberInputs = 6;
        final int inputSize =4;

        Matrix input = new Matrix(inputSize, numberInputs, index -> (random.nextDouble()));
        Matrix weights = new Matrix(numberNeurons, inputSize, index -> (random.nextGaussian()));
        Matrix biases = new Matrix(numberNeurons, 1, index -> (random.nextGaussian()));

        Matrix result1 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row));
        Matrix result2 = weights.multiply(input).modify((row, col, value) -> value + biases.get(row)).modify(value -> value > 0 ? value : 0 );

        result2.forEach((index, value) -> {
            double originalValue = result1.get(index);
            if(originalValue > 0){
                assertTrue(Math.abs(originalValue - value) <  0.000001);
            }
            else{
                assertTrue(Math.abs(value) <  0.000001);
            }
        });
    }


}
