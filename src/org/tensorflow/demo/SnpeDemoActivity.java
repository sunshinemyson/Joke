/**
@Author : SvenZ
*/
package org.tensorflow.demo;

import android.util.Log;
import android.util.Size;

public class SnpeDemoActivity extends CameraActivity {
  private final static String TAG = "Snpe_Activity";
  protected void processImage() {
    Log.i(TAG, "processImage >>>>");

    snpe_demo_main();

    Log.i(TAG, "processImage <<<<");
  }

  protected void onPreviewSizeChosen(final Size size, final int rotation) {

  }
  protected int getLayoutId() {
      return R.layout.camera_connection_fragment;
  }
  protected Size getDesiredPreviewFrameSize() {
      return new Size(640, 480);
  }

    // SvenZ:
    static {
        try {
            System.loadLibrary("snpe_demo");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Native library not found, snpe_demo won't work");
        }
    }

    protected native void snpe_demo_main();
}

