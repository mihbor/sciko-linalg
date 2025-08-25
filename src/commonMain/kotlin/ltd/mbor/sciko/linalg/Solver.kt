package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray

abstract class Solver {
  @Suppress("UNCHECKED_CAST")
  inline fun <reified D: Dimension> solve(b: MultiArray<Double, out D>): MultiArray<Double, D> {
    return when(D::class) {
      D1::class -> solveVector(b as MultiArray<Double, D1>) as MultiArray<Double, D>
      D2::class -> solveMatrix(b as MultiArray<Double, D2>) as MultiArray<Double, D>
      else -> throw IllegalArgumentException("Dimension ${D::class} not supported")
    }
  }
  abstract fun solveVector(b: RealVector): RealVector
  abstract fun solveMatrix(b: RealMatrix): RealMatrix
}