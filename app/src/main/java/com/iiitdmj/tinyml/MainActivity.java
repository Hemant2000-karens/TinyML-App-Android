package com.iiitdmj.tinyml;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};
    private static final int IMAGE_SIZE = 224;
    private static final int NUM_CLASSES = 8; // Number of classes for your model
    private static final String[] LABELS = {"Circle", "Square", "Rectangle", "Kite", "Parallelogram", "Rhombus", "Trapezoid", "Triangle"}; // Your class labels

    private ExecutorService cameraExecutor;
    private TextView resultTextView;
    private PreviewView previewView;
    private Interpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.resultTextView);
        previewView = findViewById(R.id.previewView);
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Initialize TFLite interpreter
        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException("Error loading model", e);
        }

        // Request camera permissions
        ActivityResultLauncher<String[]> requestPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestMultiplePermissions(),
                permissions -> {
                    Boolean cameraGranted = permissions.get(Manifest.permission.CAMERA);
                    if (cameraGranted != null && cameraGranted) {
                        startCamera();
                    } else {
                        resultTextView.setText(R.string.camera_permission_is_required);
                    }
                });

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS);
        }
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalyzer = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalyzer.setAnalyzer(cameraExecutor, new YourImageAnalyzer(result -> runOnUiThread(() -> resultTextView.setText(result))));

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer);
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException("Camera initialization failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetManager assetManager = getAssets();
        return assetManager.openFd("shape_classification_model.tflite").createInputStream().getChannel().map(FileChannel.MapMode.READ_ONLY, 0, assetManager.openFd("shape_classification_model.tflite").getDeclaredLength());
    }

    private class YourImageAnalyzer implements ImageAnalysis.Analyzer {
        private final OnResultListener onResultListener;

        YourImageAnalyzer(OnResultListener onResultListener) {
            this.onResultListener = onResultListener;
        }

        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            Bitmap bitmap = imageProxyToBitmap(imageProxy);
            String result = runInference(bitmap);
            onResultListener.onResult(result);
            imageProxy.close();
        }

        private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
            ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
            ByteBuffer yBuffer = planes[0].getBuffer(); // Y
            ByteBuffer uBuffer = planes[1].getBuffer(); // U
            ByteBuffer vBuffer = planes[2].getBuffer(); // V

            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            byte[] nv21 = new byte[ySize + uSize + vSize];

            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);

            Bitmap bitmap = Bitmap.createBitmap(imageProxy.getWidth(), imageProxy.getHeight(), Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(nv21));
            return Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, false);
        }

        private String runInference(Bitmap bitmap) {
            // Convert bitmap to TensorImage and apply normalization
            TensorImage tensorImage = new TensorImage();
            tensorImage.load(bitmap);

            // Create ImageProcessor with NormalizeOp
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new NormalizeOp(0, 255)) // Normalization parameters
                    .build();

            TensorImage processedImage = imageProcessor.process(tensorImage);

            // Prepare input data
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 4);
            inputBuffer.order(ByteOrder.nativeOrder());
            processedImage.getBuffer();

            // Prepare output data
            float[][] output = new float[1][NUM_CLASSES];
            interpreter.run(inputBuffer, output);
            return getTopLabel(output[0]);
        }

        private String getTopLabel(float[] outputScores) {
            int maxIndex = -1;
            float maxScore = -1;
            for (int i = 0; i < outputScores.length; i++) {
                if (outputScores[i] > maxScore) {
                    maxScore = outputScores[i];
                    maxIndex = i;
                }
            }
            return LABELS[maxIndex];
        }
    }

    private interface OnResultListener {
        void onResult(String result);
    }
}