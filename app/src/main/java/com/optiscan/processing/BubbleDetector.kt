package com.optiscan.processing

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import com.optiscan.processing.models.BubbleState
import com.optiscan.processing.models.DetectedBubble
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class BubbleDetector @Inject constructor() {

    companion object {
        private const val TAG = "BubbleDetector"

        const val SHEET_WIDTH = 800
        const val SHEET_HEIGHT = 1100

        const val GRID_START_Y = 240
        const val GRID_MARGIN_X = 30
        const val GRID_BOTTOM_MARGIN = 30
        const val NUM_OPTIONS = 5
        const val BUBBLE_DIAMETER = 22
        const val MAX_SINGLE_COL_ROWS = 30

        // Değerler: balonun, kendi satırındaki medyan balondan ne kadar koyu olduğu (0-1)
        private const val FILL_THRESHOLD = 0.18f
        private const val AMBIGUOUS_THRESHOLD = 0.10f

        private const val FRAME_X = 20.0
        private const val FRAME_Y = 20.0
        private const val FRAME_W = 760.0
        private const val FRAME_H = 1060.0
        private const val MEASURE_RADIUS = 6

        private val TOP_V = doubleArrayOf(
            0.0950, 0.1238, 0.1526, 0.1820, 0.2108,
            0.2396, 0.2684, 0.2971, 0.3266, 0.3554
        )

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

    fun computeLayout(questionCount: Int): GridLayout {
        return GridLayout(
            useSecondCol = true,
            col1Count = 20, col2Count = 20,
            rowHeight = 20, optSpacing = 20,
            col1X = 130, col2X = 310,
            questionNumX1 = 100, questionNumX2 = 280
        )
    }

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
        val background = Mat()
        val normalized = Mat()
        var bubbles: List<DetectedBubble> = emptyList()

        try {
            Imgproc.cvtColor(sheetMat, gray, Imgproc.COLOR_RGBA2GRAY)

            // Işık/gölge düzleştirme: görüntüyü bulanık arka planına böl
            Imgproc.GaussianBlur(gray, background, Size(71.0, 71.0), 0.0)
            Core.divide(gray, background, normalized, 255.0)

            bubbles = measureAll(normalized, questionCount)
            Log.d(TAG, "detectBubbles: qCount=$questionCount, detected=${bubbles.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Bubble detection failed: ${e.message}", e)
        } finally {
            gray.release()
            background.release()
            normalized.release()
        }

        return bubbles
    }

    private fun measureAll(norm: Mat, questionCount: Int): List<DetectedBubble> {
        val bubbles = mutableListOf<DetectedBubble>()
        val n = questionCount.coerceIn(1, 100)

        for (q in 0 until n) {
            val (col, v) = questionPosition(q)
            val cy = (FRAME_Y + v * FRAME_H).toInt()

            val xs = IntArray(NUM_OPTIONS) { (FRAME_X + COL_U[col][it] * FRAME_W).toInt() }
            val darkness = FloatArray(NUM_OPTIONS) { measureDarkness(norm, xs[it], cy) }

            val baseline = darkness.sorted()[NUM_OPTIONS / 2]

            for (opt in 0 until NUM_OPTIONS) {
                val fill = (darkness[opt] - baseline).coerceIn(0f, 1f)
                bubbles.add(DetectedBubble(q, opt, classifyFill(fill), fill, xs[opt], cy))
            }
        }
        return bubbles
    }

    private fun classifyFill(fill: Float) = when {
        fill >= FILL_THRESHOLD -> BubbleState.FILLED
        fill >= AMBIGUOUS_THRESHOLD -> BubbleState.AMBIGUOUS
        else -> BubbleState.EMPTY
    }

    private fun measureDarkness(norm: Mat, cx: Int, cy: Int): Float {
        val r = MEASURE_RADIUS
        val x1 = (cx - r).coerceAtLeast(0)
        val y1 = (cy - r).coerceAtLeast(0)
        val x2 = (cx + r).coerceAtMost(norm.cols() - 1)
        val y2 = (cy + r).coerceAtMost(norm.rows() - 1)
        if (x2 <= x1 || y2 <= y1) return 0f

        val roi = norm.submat(Rect(x1, y1, x2 - x1, y2 - y1))
        val mask = Mat.zeros(roi.size(), CvType.CV_8UC1)
        Imgproc.circle(
            mask,
            Point((roi.cols() / 2).toDouble(), (roi.rows() / 2).toDouble()),
            r, Scalar(255.0), -1
        )
        val mean = Core.mean(roi, mask).`val`[0]
        roi.release()
        mask.release()

        return (1.0 - mean / 255.0).toFloat().coerceIn(0f, 1f)
    }

    fun resolveAnswers(bubbles: List<DetectedBubble>, questionCount: Int): List<String> {
        val answers = MutableList(questionCount) { "" }
        val optionLabels = listOf("A", "B", "C", "D", "E")

        for (q in 0 until questionCount) {
            val filled = bubbles.filter { it.questionIndex == q && it.state == BubbleState.FILLED }

            when {
                filled.size == 1 -> {
                    answers[q] = optionLabels.getOrElse(filled[0].optionIndex) { "" }
                }
                filled.size > 1 -> {
                    val sorted = filled.sortedByDescending { it.fillPercentage }
                    if (sorted[0].fillPercentage > sorted[1].fillPercentage * 1.5f) {
                        answers[q] = optionLabels.getOrElse(sorted[0].optionIndex) { "" }
                    }
                }
            }
        }

        return answers
    }
}
