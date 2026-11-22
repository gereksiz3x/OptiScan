package com.optiscan.processing

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * ZipGrade formu için perspektif düzeltme.
 * 4 köşedeki büyük siyah karelerin merkezleri 800x1100 çıktıda
 * (20,20) (780,20) (780,1080) (20,1080) noktalarına eşlenir.
 */
@Singleton
class PerspectiveTransformer @Inject constructor() {

    companion object {
        private const val TAG = "PerspectiveTransformer"
        private const val OUTPUT_WIDTH = 800
        private const val OUTPUT_HEIGHT = 1100
        private const val OFFSET = 20.0
    }

    data class TransformResult(
        val bitmap: Bitmap?,
        val corners: Array<Point>?,
        val isSuccess: Boolean,
        val confidence: Float = 1f
    )

    fun transform(source: Bitmap): TransformResult {
        val srcMat = Mat()
        Utils.bitmapToMat(source, srcMat)

        return try {
            val enhanced = enhanceContrast(srcMat)
            val gray = Mat()
            Imgproc.cvtColor(enhanced, gray, Imgproc.COLOR_RGBA2GRAY)

            val corners = detectMarkers(gray, srcMat.cols(), srcMat.rows())
            gray.release()
            enhanced.release()

            if (corners == null) {
                Log.w(TAG, "Köşe işaretleri bulunamadı")
                return TransformResult(null, null, false)
            }

            val warped = warp(srcMat, corners)
            val bitmap = matToBitmap(warped)
            warped.release()
            TransformResult(bitmap, corners, true, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Transform failed: ${e.message}", e)
            TransformResult(null, null, false)
        } finally {
            srcMat.release()
        }
    }

    private fun enhanceContrast(src: Mat): Mat {
        val rgb = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        val lab = Mat()
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        rgb.release()

        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(channels[0], channels[0])
        Core.merge(channels, lab)
        channels.forEach { it.release() }

        val back = Mat()
        Imgproc.cvtColor(lab, back, Imgproc.COLOR_Lab2RGB)
        lab.release()
        val rgba = Mat()
        Imgproc.cvtColor(back, rgba, Imgproc.COLOR_RGB2RGBA)
        back.release()
        return rgba
    }

    private fun detectMarkers(gray: Mat, imgW: Int, imgH: Int): Array<Point>? {
        val imageArea = imgW.toDouble() * imgH.toDouble()
        val minArea = imageArea * 0.00003
        val maxArea = imageArea * 0.02

        val otsu = Mat()
        Imgproc.threshold(gray, otsu, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
        val r1 = findMarkers(otsu, imgW, imgH, minArea, maxArea, imageArea)
        otsu.release()
        if (r1 != null) return r1

        val a1 = Mat()
        Imgproc.adaptiveThreshold(gray, a1, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV, 31, 12.0)
        val r2 = findMarkers(a1, imgW, imgH, minArea, maxArea, imageArea)
        a1.release()
        if (r2 != null) return r2

        val a2 = Mat()
        Imgproc.adaptiveThreshold(gray, a2, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV, 51, 15.0)
        val r3 = findMarkers(a2, imgW, imgH, minArea, maxArea, imageArea)
        a2.release()
        if (r3 != null) return r3

        val fixed = Mat()
        Imgproc.threshold(gray, fixed, 80.0, 255.0, Imgproc.THRESH_BINARY_INV)
        val r4 = findMarkers(fixed, imgW, imgH, minArea, maxArea, imageArea)
        fixed.release()
        return r4
    }

    private fun findMarkers(
        binary: Mat, imgW: Int, imgH: Int,
        minArea: Double, maxArea: Double, imageArea: Double
    ): Array<Point>? {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()

        data class Cand(val cx: Double, val cy: Double, val area: Double)
        val candidates = mutableListOf<Cand>()

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < minArea || area > maxArea) continue

            val rect = Imgproc.boundingRect(contour)
            val aspect = rect.width.toDouble() / rect.height.toDouble()
            if (aspect < 0.7 || aspect > 1.4) continue

            val extent = area / (rect.width.toDouble() * rect.height.toDouble())
            if (extent < 0.75) continue

            candidates.add(Cand(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0, area))
        }
        contours.forEach { it.release() }

        if (candidates.size < 4) return null

        // Büyük köşe kareleri (15.5pt) küçük ara karelerden (9.5pt) büyüktür.
        // En büyük alanın %55'inden küçük olanları ele.
        val maxCandArea = candidates.maxOf { it.area }
        val big = candidates.filter { it.area >= maxCandArea * 0.55 }
        if (big.size < 4) return null

        val imgCorners = arrayOf(
            Point(0.0, 0.0),
            Point(imgW.toDouble(), 0.0),
            Point(imgW.toDouble(), imgH.toDouble()),
            Point(0.0, imgH.toDouble())
        )

        val remaining = big.toMutableList()
        val found = Array<Point?>(4) { null }
        for (idx in imgCorners.indices) {
            val best = remaining.minByOrNull { distance(Point(it.cx, it.cy), imgCorners[idx]) }
                ?: return null
            found[idx] = Point(best.cx, best.cy)
            remaining.remove(best)
        }

        @Suppress("UNCHECKED_CAST")
        val result = found as Array<Point>

        val quadArea = quadArea(result)
        if (quadArea < imageArea * 0.05 || quadArea > imageArea * 0.98) return null

        val d01 = distance(result[0], result[1])
        val d12 = distance(result[1], result[2])
        val d23 = distance(result[2], result[3])
        val d30 = distance(result[3], result[0])
        if (d01 > 0 && d23 > 0 && min(d01, d23) / max(d01, d23) < 0.6) return null
        if (d12 > 0 && d30 > 0 && min(d12, d30) / max(d12, d30) < 0.6) return null

        return result
    }

    private fun warp(src: Mat, c: Array<Point>): Mat {
        val srcPts = MatOfPoint2f(c[0], c[1], c[2], c[3])
        val dstPts = MatOfPoint2f(
            Point(OFFSET, OFFSET),
            Point(OUTPUT_WIDTH - OFFSET, OFFSET),
            Point(OUTPUT_WIDTH - OFFSET, OUTPUT_HEIGHT - OFFSET),
            Point(OFFSET, OUTPUT_HEIGHT - OFFSET)
        )
        val m = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val warped = Mat()
        Imgproc.warpPerspective(
            src, warped, m,
            Size(OUTPUT_WIDTH.toDouble(), OUTPUT_HEIGHT.toDouble()),
            Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE
        )
        srcPts.release(); dstPts.release(); m.release()
        return warped
    }

    private fun quadArea(pts: Array<Point>): Double {
        var area = 0.0
        for (i in pts.indices) {
            val j = (i + 1) % pts.size
            area += pts[i].x * pts[j].y - pts[j].x * pts[i].y
        }
        return abs(area) / 2.0
    }

    private fun distance(a: Point, b: Point): Double =
        sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))

    private fun matToBitmap(mat: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        return bitmap
    }
}
