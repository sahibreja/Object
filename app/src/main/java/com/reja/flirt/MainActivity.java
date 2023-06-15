package com.reja.flirt;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import android.view.TextureView;
import android.widget.ImageView;

import com.reja.flirt.ml.SsdMobilenetV11Metadata1;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    private TextureView textureView;
    private List<String> labels;
    private List<Integer> colors = Arrays.asList(
            Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
            Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED
    );
    private Paint paint = new Paint();
    private ImageProcessor imageProcessor;
    private Bitmap bitmap;
    private CameraDevice cameraDevice;
    private Handler handler;
    private CameraManager cameraManager;
    private SsdMobilenetV11Metadata1 model;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
       getPermission();
        try {
            labels = FileUtil.loadLabels(this, "labels.txt");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        imageProcessor = new ImageProcessor.Builder().add(new ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build();
        try {
            model = SsdMobilenetV11Metadata1.newInstance(this);
        } catch (Exception e) {
            e.printStackTrace();
        }
        HandlerThread handlerThread = new HandlerThread("videoThread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

        imageView = findViewById(R.id.imageView);

        textureView = findViewById(R.id.textureView);

        textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {
                openCamera();
            }

            @Override
            public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {

            }

            @Override
            public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surfaceTexture) {
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surfaceTexture) {
                bitmap = textureView.getBitmap();
                TensorImage image = TensorImage.fromBitmap(bitmap);
                image = imageProcessor.process(image);
                SsdMobilenetV11Metadata1.Outputs outputs = model.process(image);
                float[] locations = outputs.getLocationsAsTensorBuffer().getFloatArray();
                float[] classes = outputs.getClassesAsTensorBuffer().getFloatArray();
                float[] scores = outputs.getScoresAsTensorBuffer().getFloatArray();
                float[] numDetections = outputs.getNumberOfDetectionsAsTensorBuffer().getFloatArray();

                Bitmap mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                Canvas canvas = new Canvas(mutable);

                int h = mutable.getHeight();
                int w = mutable.getWidth();
                paint.setTextSize(h / 15f);
                paint.setStrokeWidth(h / 85f);
                int x = 0;
                for (int index = 0; index < scores.length; index++) {
                    float fl = scores[index];
                    x = index;
                    x *= 4;
                    if (fl > 0.5) {
                        paint.setColor(colors.get(index));
                        paint.setStyle(Paint.Style.STROKE);
                        canvas.drawRect(new RectF(locations[x + 1] * w, locations[x] * h, locations[x + 3] * w, locations[x + 2] * h), paint);
                        paint.setStyle(Paint.Style.FILL);
                        canvas.drawText(labels.get((int) classes[index]) + " " + Float.toString(fl), locations[x + 1] * w, locations[x] * h, paint);
                    }
                }
                imageView.setImageBitmap(mutable);

            }
        });
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        model.close();
    }

    private void openCamera() {
        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                // TODO: Consider calling
                //    ActivityCompat#requestPermissions
                // here to request the missing permissions, and then overriding
                //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                //                                          int[] grantResults)
                // to handle the case where the user grants the permission. See the documentation
                // for ActivityCompat#requestPermissions for more details.
                return;
            }
            cameraManager.openCamera(cameraManager.getCameraIdList()[0], new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    cameraDevice = camera;

                    SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
                    @SuppressLint("Recycle") Surface surface = new Surface(surfaceTexture);


                    try {
                        CaptureRequest.Builder captureRequest=cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        captureRequest.addTarget(surface);

                        cameraDevice.createCaptureSession(Collections.singletonList(surface), new CameraCaptureSession.StateCallback() {
                            @Override
                            public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                                try {
                                    cameraCaptureSession.setRepeatingRequest(captureRequest.build(), null, null);
                                } catch (CameraAccessException e) {
                                    throw new RuntimeException(e);
                                }
                            }

                            @Override
                            public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {

                            }
                        },handler);


                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {

                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {

                }
            }, handler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void getPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 101);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission();
        }
    }


}