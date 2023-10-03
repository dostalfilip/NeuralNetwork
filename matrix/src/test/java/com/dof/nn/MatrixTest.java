package com.dof.nn;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MatrixTest {
    @Test
    void test(){
        Matrix matrix = new Matrix(3, 4, i->i*2);
        String result = matrix.toString();

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
