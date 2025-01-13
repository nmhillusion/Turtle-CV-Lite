package tech.nmhillusion.turtlecvlite

import android.graphics.Point
import org.opencv.core.Mat
import kotlin.math.pow
import kotlin.math.sqrt


/**
 * date: 2025-01-13
 * author: nmhillusion
 */

/**
 * created by: nmhillusion
 * <p>
 * created date: 2025-01-13
 */
class BlinkEyeDetector {
    private val EYE_AR_THRESH: Int = 5 // Threshold for EAR
    private val EYE_AR_CONSEC_FRAMES: Int = 3 // Number of frames to count as a blink

    private var blinkCounter = 0 // Counter for consecutive frames with EAR below threshold
    private var totalBlinks = 0 // Total number of blinks detected

    fun detectBlink(frame: Mat): Boolean {
        // Assuming you have already detected landmarks and have their coordinates
        val leftEye: Array<Point> = arrayOf<Point>(
            Point(36, 0),
            Point(37, 1),
            Point(38, 2),
            Point(39, 3),
            Point(40, 4),
            Point(41, 5)
        )
        val rightEye: Array<Point> = arrayOf<Point>(
            Point(42, 0),
            Point(43, 1),
            Point(44, 2),
            Point(45, 3),
            Point(46, 4),
            Point(47, 5)
        )

        val earLeft = calculateEAR(leftEye)
        val earRight = calculateEAR(rightEye)

        val ear = (earLeft + earRight) / 2.0 // Average EAR for both eyes

        // Check if EAR is below threshold
        if (ear < EYE_AR_THRESH) {
            blinkCounter++
            if (blinkCounter >= EYE_AR_CONSEC_FRAMES) {
                totalBlinks++
//                updateTextResult("Blinks detected: $totalBlinks")
                return true // Blink detected
            }
        } else {
            blinkCounter = 0 // Reset counter if eyes are open
        }

        return false // No blink detected
    }

    private fun calculateEAR(eyePoints: Array<Point>): Double {
        val A = distance(eyePoints[1], eyePoints[5]) // Vertical distance
        val B = distance(eyePoints[2], eyePoints[4]) // Vertical distance
        val C = distance(eyePoints[0], eyePoints[3]) // Horizontal distance

        return (A + B) / (2.0 * C) // Calculate EAR
    }

    private fun distance(p1: Point, p2: Point): Double {
        return sqrt((p1.x - p2.x).toDouble().pow(2.0) + (p1.y - p2.y).toDouble().pow(2.0))
    }

}