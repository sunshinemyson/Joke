/**
@Author : SvenZ
*/
package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.media.ImageReader;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class SnpeDemoActivity extends CameraActivity {
    private final static String TAG = "Snpe_Activity";



    private Bitmap rgbFrameBitmap = null; // todo, I don't need this
    // SvenZ:
    static {
        try {
            System.loadLibrary("snpe_demo");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Native library not found, snpe_demo won't work");
        }
    }

    private String dlcFilePath;
    private String dspRuntimePath;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        dlcFilePath = copyAssets()[0];
        Log.i(TAG, "Asset copy done");

        dspRuntimePath = getApplication().getApplicationInfo().nativeLibraryDir;
        Log.i(TAG, "native library dir=" + dspRuntimePath);
        rgbFrameBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        //Log.i(TAG, "Image read");
        super.onImageAvailable(reader);
    }

    @Override
    protected void processImage() {
        Log.i(TAG, "processImage >>>>");
        // todo, workflow refine
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        Log.i(TAG, "frame bmp" + rgbFrameBitmap.getWidth() + rgbFrameBitmap.getHeight());

        snpe_demo_main(dlcFilePath,dspRuntimePath,rgbFrameBitmap);

        Log.i(TAG, "processImage <<<<");
        readyForNextImage();        // Go ahead
    }

    @Override
    protected void onPreviewSizeChosen(final Size size, final int rotation) {
        Log.i(TAG, "PreviewSizeChosed:<" + size.toString() +"> rotation=" + rotation);
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return new Size(640, 480);
    }

    private String[] copyAssets() {
        AssetManager assetManager = getAssets();
        String[] files = null;
        try {
            files = assetManager.list("");
            for (String fname : files) {
                Log.i(TAG, fname);
            }
        } catch (IOException e) {
            Log.e("tag", "Failed to get asset file list.", e);
        }

        String[] absolutePath =new String[files.length];
        if (files != null) for (String filename : files) {

            if (!filename.endsWith("dlc")) continue;

            InputStream in = null;
            OutputStream out = null;
            try {
                in = assetManager.open(filename);
                File outFile = new File(getExternalFilesDir(null), filename);
                Log.i(TAG, "start copy " +filename + "to " + outFile.getAbsolutePath());
                // todo: remove hardcode
                absolutePath[0] = outFile.getAbsolutePath();
                out = new FileOutputStream(outFile);
                copyFile(in, out);
            } catch(IOException e) {
                Log.e("tag", "Failed to copy asset file: " + filename, e);
            }
            finally {
                if (in != null) {
                    try {
                        in.close();
                    } catch (IOException e) {
                        // NOOP
                    }
                }
                if (out != null) {
                    try {
                        out.close();
                    } catch (IOException e) {
                        // NOOP
                    }
                }
            }
        }

        return absolutePath;
    }

    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }

     protected native void snpe_demo_main(String dlcFilePath, String dspRuntimeLibPath, Bitmap cameraFrame);
}


