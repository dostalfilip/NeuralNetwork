package com.dof.nn;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MatrixTest {

    @Test
    void testMultiply(){
        Matrix m1 = new Matrix(2, 3, i-> i);
        Matrix m2 = new Matrix(3, 2, i-> i);

        System.out.println(m1);
        System.out.println(m2);

        Matrix result = m1.multiply(m2);
    }

    @Test
    void testEquals(){
        Matrix m1 = new Matrix(3, 4, i-> 2 * (i-6));
        Matrix m2 = new Matrix(3, 4, i-> 2 * (i-6));
        Matrix m3 = new Matrix(3, 4, i-> 2 * (i-6.2));

        assertTrue(m1.equals(m2));
        assertFalse(m1.equals(m3));
    }

    @Test
    void testMultiplyDouble(){
        Matrix m = new Matrix(3, 4, i-> 2.5 * (i-6));

        double x = 0.5;
        Matrix expected = new Matrix(3, 4, i-> x * 2.5 * (i-6));

        Matrix result = m.apply((index, value) -> x * value);

        assertTrue(Math.abs(expected.get(0) + 7.50000) < 0.0001);
        assertEquals(expected, result);
    }

    @Test
    void testAddMatrices(){
        Matrix m1 = new Matrix(2, 2, i->i);
        Matrix m2 = new Matrix(2, 2, i->i * 1.5);
        Matrix expected = new Matrix(2, 2, i->i * 2.5);

        Matrix result = m1.apply((index, value) -> value + m2.get(index));

        assertTrue(result.equals(expected));
    }

    @Test
    void testToString(){
        Matrix matrix = new Matrix(3, 4, i->i*2);
        String result = matrix.toString();

        System.out.println(result);

        int index = 0;
        double[] expected = new double[12];

        for(int i = 0; i < expected.length; i++){
            expected[i] = i*2;
        }

        var  rawRows = result.split("\n");

        assertEquals(3, rawRows.length);

        for(var rawRow : rawRows){
            var  rawCols = rawRow.split("\\s+");

            for(var rawCol : rawCols){

                if(rawCol.length() == 0){
                    continue;
                }

                assertTrue(Math.abs(expected[index] - Double.valueOf(rawCol)) < 0.00001);

                index++;
            }

        }
    }

}
