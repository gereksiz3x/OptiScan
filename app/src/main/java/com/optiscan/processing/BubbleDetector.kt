package com.optiscan.processing

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import com.optiscan.processing.models.BubbleState
import com.optiscan.processing.models.DetectedBubble
import javax.inject.Inject
import javax.inject.Singleton

/**
 * ZipGrade 100 soruluk form için balon okuyucu.
 * Düzeltilmiş görüntü 800x1100, köşe kare merkezleri (20,20)-(780,1080).
 * Sütunlar: A = 1-20, B = 21-40, C = 41-70, D = 71-100.
 */
@Singleton
class BubbleDetector @Inject constructor() {

    companion object {
        private const val TAG = "BubbleDetector"

        const val SHEET_WIDTH = 800
        const val SHEET_HEIGHT = 1100

        // Diğer dosyalar kullanıyor olabilir, derleme hatası olmasın diye duruyor
        const val GRID_START_Y = 240
        const val GRID_MARGIN_X = 30
        const val GRID_BOTTOM_MARGIN = 30
        const val NUM_OPTIONS = 5
        const val BUBBLE_DIAMETER = 22
        const val MAX_SINGLE_COL_ROWS = 30

        private const val FILL_THRESHOLD = 0.32f
        private const val AMBIGUOUS_THRESHOLD = 0.15f

        private const val FRAME_X = 20.0
        private const val FRAME_Y = 20.0
        private const val FRAME_W = 760.0
        private const val FRAME_H = 1060.0
        private const val MEASURE_RADIUS = 8

        // Üst blok (C ve D sütunlarında 41-50 / 71-80)
        private val TOP_V = doubleArrayOf(
            0.0950, 0.1238, 0.1526, 0.1820, 0.2108,
            0.2396, 0.2684, 0.2971, 0.3266, 0.3554
        )

        // A/B sütunları (20 satır) ve C/D sütunlarının alt 20 satırı
        private val LOWER_V = doubleArrayOf(
            0.4018, 0.4306, 0.4594, 0.4888, 0.5176,
            0.5463, 0.5751, 0.6039, 0.6333, 0.6621,
            0.7073, 0.7367, 0.7655, 0.7949, 0.8237,
            0.8525, 0.8819, 0.9107, 0.9401, 0.9689
        )

        private val COL_U = arrayOf(
            doubleArrayOf(0.1724, 0.2061, 0.2398, 0.2744, 0.3090),
            doubleArrayOf(0.3864, 0.4210, 0.4547, 0.4893, 0.5230),
            doubleArrayOf(0.5995, 0.6332, 0.6678, 0.7015, 0.7361),
            doubleArrayOf(0.8126, 0.8463, 0.8800, 0.9146, 0.9483)
        )
    }

    data class GridLayout(
        val useSecondCol: Boolean,
        val col1Count: Int,
        val col2Count: Int,
        val rowHeight: Int,
        val optSpacing: Int,
        val col1X: Int,
        val col2X: Int,
        val questionNumX1: Int,
        val questionNumX2: Int
    )

    // Eski çağrılar için duruyor, artık okumada kullanılmıyor
    fun computeLayout(questionCount: Int): GridLayout {
        return GridLayout(
            useSecondCol = true,
            col1Count = 20, col2Count = 20,
            rowHeight = 20, optSpacing = 20,
            col1X = 130, col2X = 310,
            questionNumX1 = 100, questionNumX2 = 280
        )
    }

    /** Soru indeksi (0 tabanlı) → balon merkezi y ve sütun */
    private fun questionPosition(q: Int): Pair<Int, Double> {
        return when {
            q < 20 -> Pair(0, LOWER_V[q])
            q < 40 -> Pair(1, LOWER_V[q - 20])
            q < 70 -> {
                val r = q - 40
                Pair(2, if (r < 10) TOP_V[r] else LOWER_V[r - 10])
            }
            else -> {
                val r = q - 70
                Pair(3, if (r < 10) TOP_V[r] else LOWER_V[r - 10])
            }
        }
    }

    fun detectBubbles(sheetMat: Mat, questionCount: Int): List<DetectedBubble> {
        val gray = Mat()
        val detectedBubbles = mutableListOf<DetectedBubble>()

        try {
            Imgproc.cvtColor(sheetMat, gray, Imgproc.COLOR_RGBA2GRAY)

            val clahe = Imgproc.createCLAHE(3.0, Size(8.0, 8.0))
            val enhanced = Mat()
            clahe.apply(gray, enhanced)

            val threshResults = mutableListOf<Pair<Mat, List<DetectedBubble>>>()
            val paramSets = listOf(
                Pair(15, 8.0),
                Pair(21, 10.0),
                Pair(31, 12.0)
            )

            for ((blockSize, c) in paramSets) {
                val thresh = Mat()
                Imgproc.adaptiveThreshold(
                    enhanced, thresh, 255.0,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY_INV,
                    blockSize, c
                )
                val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
                Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_OPEN, kernel)
                kernel.release()

                val bubbles = detectBubblesFromThresh(thresh, questionCount)
                threshResults.add(Pair(thresh, bubbles))
            }

            enhanced.release()

            val best = threshResults.maxByOrNull { (_, bubbles) ->
                val answers = resolveAnswers(bubbles, questionCount)
                val answeredCount = answers.count { it.isNotBlank() }
                val ambiguousCount = bubbles.count { it.state == BubbleState.AMBIGUOUS }
                answeredCount * 10 - ambiguousCount
            }

            threshResults.forEach { (mat, _) -> mat.release() }

            if (best != null) detectedBubbles.addAll(best.second)

            Log.d(TAG, "detectBubbles: qCount=$questionCount, detected=${detectedBubbles.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Bubble detection failed: ${e.message}", e)
        } finally {
            gray.release()
        }

        return detectedBubbles
    }

    private fun detectBubblesFromThresh(thresh: Mat, questionCount: Int): List<DetectedBubble> {
        val bubbles = mutableListOf<DetectedBubble>()
        val n = questionCount.coerceIn(1, 100)

        for (q in 0 until n) {
            val (col, v) = questionPosition(q)
            val cy = (FRAME_Y + v * FRAME_H).toInt()

            for (opt in 0 until NUM_OPTIONS) {
                val cx = (FRAME_X + COL_U[col][opt] * FRAME_W).toInt()
                val fillPct = measureBubbleFill(thresh, cx, cy, MEASURE_RADIUS)
                bubbles.add(DetectedBubble(q, opt, classifyFill(fillPct), fillPct, cx, cy))
            }
        }
        return bubbles
    }

    private fun classifyFill(fillPct: Float) = when {
        fillPct >= FILL_THRESHOLD -> BubbleState.FILLED
        fillPct >= AMBIGUOUS_THRESHOLD -> BubbleState.AMBIGUOUS
        else -> BubbleState.EMPTY
    }

    private fun measureBubbleFill(threshMat: Mat, cx: Int, cy: Int, radius: Int): Float {
        val r = radius.coerceAtLeast(4)
        val x1 = (cx - r).coerceAtLeast(0)
        val y1 = (cy - r).coerceAtLeast(0)
        val x2 = (cx + r).coerceAtMost(threshMat.cols() - 1)
        val y2 = (cy + r).coerceAtMost(threshMat.rows() - 1)

        if (x2 <= x1 || y2 <= y1) return 0f

        val roi = threshMat.submat(Rect(x1, y1, x2 - x1, y2 - y1))
        val mask = Mat.zeros(roi.size(), CvType.CV_8UC1)
        Imgproc.circle(
            mask,
            Point((roi.cols() / 2).toDouble(), (roi.rows() / 2).toDouble()),
            r, Scalar(255.0), -1
        )

        val masked = Mat()
        Core.bitwise_and(roi, mask, masked)

        val darkPixels = Core.countNonZero(masked)
        val totalPixels = Core.countNonZero(mask)

        roi.release()
        mask.release()
        masked.release()

        return if (totalPixels == 0) 0f else darkPixels.toFloat() / totalPixels.toFloat()
    }

    fun resolveAnswers(bubbles: List<DetectedBubble>, questionCount: Int): List<String> {
        val answers = MutableList(questionCount) { "" }
        val optionLabels = listOf("A", "B", "C", "D", "E")

        for (q in 0 until questionCount) {
            val questionBubbles = bubbles.filter { it.questionIndex == q }
            val filledBubbles = questionBubbles.filter { it.state == BubbleState.FILLED }

            when {
                filledBubbles.size == 1 -> {
                    answers[q] = optionLabels.getOrElse(filledBubbles[0].optionIndex) { "" }
                }
                filledBubbles.size > 1 -> {
                    val sorted = filledBubbles.sortedByDescending { it.fillPercentage }
                    if (sorted[0].fillPercentage > sorted[1].fillPercentage * 1.5f) {
                        answers[q] = optionLabels.getOrElse(sorted[0].optionIndex) { "" }
                    } else {
                        answers[q] = ""
                    }
                }
                else -> {
                    val ambiguous = questionBubbles.filter { it.state == BubbleState.AMBIGUOUS }
                    if (ambiguous.size == 1) {
                        answers[q] = optionLabels.getOrElse(ambiguous[0].optionIndex) { "" }
                    } else {
                        answers[q] = ""
                    }
                }
            }
        }

        return answers
    }
}
