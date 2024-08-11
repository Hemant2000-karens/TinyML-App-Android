package com.iiitdmj.tinyml;

import static androidx.camera.core.resolutionselector.AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY;

import android.os.Bundle;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.widget.TextView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_PERMISSIONS = 10;
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
        ActivityResultLauncher<String[]> requestPermissionLauncher = registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), permissions -> {
            Boolean cameraGranted = permissions.get(Manifest.permission.CAMERA);
            if (cameraGranted != null && cameraGranted) {
                startCamera();
            } else {
                // Permission request was denied.
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
                        .setResolutionSelector(new ResolutionSelector.Builder()
                                .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                                .setAspectRatioStrategy(RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                                .build())
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
        try (FileInputStream fileInputStream = new FileInputStream(getAssets().openFd("shape_classification_model.tflite").getFileDescriptor())) {
            FileChannel fileChannel = fileInputStream.getChannel();
            long startOffset = getAssets().openFd("shape_classification_model.tflite").getStartOffset();
            long declaredLength = getAssets().openFd("shape_classification_model.tflite").getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    public class YourImageAnalyzer implements ImageAnalysis.Analyzer {
        private final OnResultListener onResultListener;

        YourImageAnalyzer(OnResultListener onResultListener) {
            this.onResultListener = onResultListener;
        }

        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            Bitmap bitmap = imageProxyToBitmap(imageProxy);
            if (bitmap != null) {
                String result = runInference(bitmap);
                onResultListener.onResult(result);
            }
            imageProxy.close();
        }

        private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
            ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
            ByteBuffer yBuffer = planes[0].getBuffer(); // Y

            // Convert the YUV image to grayscale
            byte[] nv21 = new byte[yBuffer.remaining()];
            yBuffer.get(nv21);

            // Create a grayscale bitmap
            Bitmap grayscaleBitmap = Bitmap.createBitmap(imageProxy.getWidth(), imageProxy.getHeight(), Bitmap.Config.ARGB_8888);

            int[] pixels = new int[imageProxy.getWidth() * imageProxy.getHeight()];
            for (int i = 0; i < nv21.length; i++) {
                int luminance = nv21[i] & 0xFF;
                pixels[i] = Color.rgb(luminance, luminance, luminance);
            }

            grayscaleBitmap.setPixels(pixels, 0, imageProxy.getWidth(), 0, 0, imageProxy.getWidth(), imageProxy.getHeight());

            // Resize and center-crop the image to maintain aspect ratio
            return Bitmap.createScaledBitmap(grayscaleBitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        }

        private String runInference(Bitmap bitmap) {
            // Convert the bitmap to a ByteBuffer
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 4);
            inputBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

            // Normalize the pixel values to [-1, 1] range
            for (int pixelValue : intValues) {
                float normalizedValue = ((pixelValue & 0xFF) / 127.5f) - 1;
                inputBuffer.putFloat(normalizedValue);
            }

            // Prepare output data
            float[][] output = new float[1][NUM_CLASSES];

            // Run model inference
            interpreter.run(inputBuffer, output);

            // Get classification result
            return getTopLabel(output[0]);
        }

        private String getTopLabel(float[] outputScores) {
            // Find the highest score and return corresponding label
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
