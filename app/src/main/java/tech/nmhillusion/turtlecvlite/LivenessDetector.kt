package tech.nmhillusion.turtlecvlite

import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.objdetect.CascadeClassifier
import org.opencv.imgproc.Imgproc

class LivenessDetector(faceCascadePath: String, eyeCascadePath: String) {

    private val faceCascade = CascadeClassifier(faceCascadePath)
    private val eyeCascade = CascadeClassifier(eyeCascadePath)

    // Function to calculate Eye Aspect Ratio (EAR)
    private fun calculateEAR(eye: MatOfPoint2f): Double {
        val eyePoints = eye.toArray()
        val A = Math.sqrt(Math.pow((eyePoints[1].x - eyePoints[5].x), 2.0) + Math.pow((eyePoints[1].y - eyePoints[5].y), 2.0))
        val B = Math.sqrt(Math.pow((eyePoints[2].x - eyePoints[4].x), 2.0) + Math.pow((eyePoints[2].y - eyePoints[4].y), 2.0))
        val C = Math.sqrt(Math.pow((eyePoints[0].x - eyePoints[3].x), 2.0) + Math.pow((eyePoints[0].y - eyePoints[3].y), 2.0))
        return (A + B) / (2.0 * C)
    }

    fun detect(frame: Mat, faceDetectedCallback: ((faceDetected: Boolean) -> Unit)?): Mat {
        val grayFrame = Mat()
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_RGBA2GRAY)

        // Detect faces
        val faces = MatOfRect()
        faceCascade.detectMultiScale(grayFrame, faces)
        var blinkDetected = false

        var faceDetected = false
        for (face in faces.toArray()) {
            Imgproc.rectangle(frame, face.tl(), face.br(), Scalar(0.0, 255.0, 0.0, 255.0), 3)

            val faceROI = grayFrame.submat(face)

            // Detect eyes
            val eyes = MatOfRect()
            eyeCascade.detectMultiScale(faceROI, eyes)
            if (eyes.toArray().isNotEmpty()) {
                faceDetected = true
            }

            for (eye in eyes.toArray()) {
                val eyeCenter = Point(face.tl().x + eye.x + eye.width * 0.5,
                    face.tl().y + eye.y + eye.height * 0.5)
                val radius = ((eye.width + eye.height) * 0.25).toInt()
                Imgproc.circle(frame, eyeCenter, radius, Scalar(255.0, 0.0, 0.0, 255.0), 3)

                // Calculate EAR
                val eyeROI = faceROI.submat(eye)
                val eyePoints = MatOfPoint2f()
                getEyePoints(eyeROI, eyePoints)
                val ear = calculateEAR(eyePoints)
                if (ear < 0.25) { // Threshold for blink detection
                    blinkDetected = true
                    break
                }
            }
        }

        faceDetectedCallback?.invoke(faceDetected)

        return frame
    }
}