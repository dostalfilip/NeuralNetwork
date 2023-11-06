package com.dof.nn.loader.test;

import com.dof.nn.core.Util;
import com.dof.nn.loader.BatchData;
import com.dof.nn.loader.Loader;
import com.dof.nn.loader.MetaData;

public class TestLoader implements Loader {
    private final MetaData metaData;

    private final int inputSize = 500;
    private final int expectedSize = 3;
    private final int numberItems;
    private int numberBatches;
    private final int batchSize;

    private int totalItemsRead;
    private int itemsRead;

    public TestLoader(int numberItems, int batchSize) {
        this.numberItems = numberItems;
        this.batchSize = batchSize;

        this.metaData = new TestMetaData();
        metaData.setNumberItems(numberItems);
        numberBatches = numberItems / batchSize;

        if (numberItems % batchSize != 0) {
            numberBatches += 1;
        }

        metaData.setNumberBatches(numberBatches);
        metaData.setInputSize(inputSize);
        metaData.setExpectedSize(expectedSize);
    }

    @Override
    public MetaData open() {
        return metaData;
    }

    @Override
    public void close() {
        totalItemsRead = 0;
    }

    @Override
    public MetaData getMetaData() {
        return metaData;
    }

    @Override
    public synchronized BatchData readBatch() {
        if (totalItemsRead == numberItems) {
            return null;
        }

        itemsRead = batchSize;
        totalItemsRead += itemsRead;

        int excessItems = totalItemsRead - numberItems;
        if (excessItems > 0) {
            totalItemsRead -= excessItems;
            itemsRead -= excessItems;
        }

        var io = Util.generateTrainingArrays(inputSize, expectedSize, itemsRead);

        var batchData = new TestBatchData();
        batchData.setInputBatch(io.getInput());
        batchData.setExpectedBatch(io.getOutput());

        metaData.setTotalItemsRead(totalItemsRead);
        metaData.setItemsRead(itemsRead);

        return batchData;
    }
}
