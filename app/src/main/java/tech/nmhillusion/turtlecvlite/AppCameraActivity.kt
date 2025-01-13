package tech.nmhillusion.turtlecvlite

import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.TextView
import org.opencv.android.CameraActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfRect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.HOGDescriptor
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream


class AppCameraActivity : CameraActivity(), CvCameraViewListener2 {
    private var livenessDetector: LivenessDetector? = null
    private val TAG = "app::AppCameraActivity"
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private var mTextResult: TextView? = null
    private var faceDetectedCount: Int = 0
    private var blinkDetectedCount: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // Set up camera view
        val cameraView: CameraBridgeViewBase = findViewById(R.id.camera_view);

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "ERROR when init for OpenCV")
        } else {
            Log.i(TAG, "Successful to init OpenCV in local app")
        }

        cameraView.visibility = SurfaceView.VISIBLE;
        cameraView.setCvCameraViewListener(this);
        mOpenCvCameraView = cameraView

        mTextResult = findViewById(R.id.textview_first)

        // Initialize LivenessDetector with cascade paths
        val loadHaarCascades = loadHaarCascades()
        livenessDetector = LivenessDetector(
            loadHaarCascades["fc"] ?: "",
            loadHaarCascades["ec"] ?: "",
        )
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null) mOpenCvCameraView?.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (mOpenCvCameraView != null) mOpenCvCameraView?.enableView()
    }

    protected override fun getCameraViewList(): List<CameraBridgeViewBase> {
        logToGui("camera is $mOpenCvCameraView")
        return if (null != mOpenCvCameraView) {
            listOf(mOpenCvCameraView as CameraBridgeViewBase)
        } else {
            listOf()
        };
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView?.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        logToGui("camera started with width = $width, height = $height")
    }

    override fun onCameraViewStopped() {
        logToGui("camera is stopped")
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val rgbaVal = inputFrame.rgba()

        logToGui("RGBA = $rgbaVal")

        val outFrame = livenessDetector?.detect(rgbaVal) { faceDetected, blinkDetected ->
            run {
                faceDetectedCount += if (faceDetected) 1 else 0
                blinkDetectedCount += if (blinkDetected) 1 else 0
            }
        }

        if (outFrame != null) {
            detectHuman(outFrame)
        }

        return outFrame ?: rgbaVal
    }

    private fun loadHaarCascades(): Map<String, String> {
        try {
            // Load face cascade
            val faceCascadeFile = File(cacheDir, "haarcascade_frontalface_default.xml")
            val faceCascadeInputStream: InputStream =
                resources.openRawResource(R.raw.haarcascade_frontalface_default)
            val faceCascadeOutputStream = FileOutputStream(faceCascadeFile)
            faceCascadeInputStream.copyTo(faceCascadeOutputStream)
            faceCascadeInputStream.close()
            faceCascadeOutputStream.close()
            val fc = faceCascadeFile.absolutePath

            // Load eye cascade
            val eyeCascadeFile = File(cacheDir, "haarcascade_eye.xml")
            val eyeCascadeInputStream: InputStream =
                resources.openRawResource(R.raw.haarcascade_eye)
            val eyeCascadeOutputStream = FileOutputStream(eyeCascadeFile)
            eyeCascadeInputStream.copyTo(eyeCascadeOutputStream)
            eyeCascadeInputStream.close()
            eyeCascadeOutputStream.close()
            val ec = eyeCascadeFile.absolutePath

            return mapOf("fc" to fc, "ec" to ec)
        } catch (ex: Exception) {
            logToGui("Error loading cascade", ex)
            return mapOf()
        }
    }

    private fun detectHuman(frame: Mat) {
        /// Convert frame to grayscale
        val grayFrame = Mat()
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_RGBA2GRAY)

        // Initialize HOG descriptor
        val hog = HOGDescriptor()
        hog.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector())

        val boxes = MatOfRect()
        val weights = MatOfDouble()

        // Detect people in the grayscale frame
        hog.detectMultiScale(grayFrame, boxes, weights)

        // Draw rectangles around detected people
        for (rect in boxes.toArray()) {
            Imgproc.rectangle(frame, rect.tl(), rect.br(), Scalar(0.0, 255.0, 255.0), 2)
        }
    }

    private fun logToGui(msg: CharSequence = "", ex: Exception? = null) {
        try {
            runOnUiThread {
                mTextResult?.text =
                    "$msg :: face detected count: $faceDetectedCount, blink detected count: $blinkDetectedCount"
            }

            if (null == ex) {
                Log.i(TAG, msg.toString())
            } else {
                Log.e(TAG, msg.toString(), ex)
            }
        } catch (ex: Exception) {
            Log.e(TAG, "Log to GUI Fail!", ex)
        }

    }

}