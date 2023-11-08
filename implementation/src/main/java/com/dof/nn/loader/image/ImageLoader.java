package com.dof.nn.loader.image;

import com.dof.nn.loader.BatchData;
import com.dof.nn.loader.Loader;
import com.dof.nn.loader.MetaData;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.nio.file.Path;

public class ImageLoader implements Loader {
    private Path imageFileName;
    private Path labelFileName;
    private int batchSize;

    private DataInputStream dsImages;
    private DataInputStream dsLabels;

    public ImageLoader(Path imageFileName, Path labelFileName, int batchSize) {
        this.imageFileName = imageFileName;
        this.labelFileName = labelFileName;
        this.batchSize = batchSize;
    }

    @Override
    public MetaData open() {
        try {
            dsImages = new DataInputStream(new FileInputStream(imageFileName.toFile()));
        } catch (Exception e) {
            throw new LoaderException("Cannot open " + imageFileName, e);
        }

        try {
            dsLabels = new DataInputStream(new FileInputStream(labelFileName.toFile()));
        } catch (Exception e) {
            throw new LoaderException("Cannot open " + labelFileName, e);
        }

        return null;
    }

    @Override
    public void close() {
        try {
            dsImages.close();
        } catch (Exception e) {
            throw new LoaderException("Cannot close" + imageFileName, e);
        }
        try {
            dsLabels.close();
        } catch (Exception e) {
            throw new LoaderException("Cannot close" + labelFileName, e);
        }
    }

    @Override
    public MetaData getMetaData() {
        return null;
    }

    @Override
    public BatchData readBatch() {
        return null;
    }
}
