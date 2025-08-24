package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.math.abs
import kotlin.math.max

class MatrixUtils {
  companion object {
    /**
     * Checks whether a matrix is symmetric.
     *
     * @param matrix Matrix to check.
     * @param eps Relative tolerance.
     * @return `true` if `matrix` is symmetric.
     * @since 3.1
     */
    fun isSymmetric(
      matrix: RealMatrix,
      eps: Double
    ): Boolean {
      return isSymmetricInternal(matrix, eps, false)
    }

    /**
     * Checks whether a matrix is symmetric, within a given relative tolerance.
     *
     * @param matrix Matrix to check.
     * @param relativeTolerance Tolerance of the symmetry check.
     * @param raiseException If `true`, an exception will be raised if
     * the matrix is not symmetric.
     * @return `true` if `matrix` is symmetric.
     * @throws NonSquareMatrixException if the matrix is not square.
     * @throws NonSymmetricMatrixException if the matrix is not symmetric.
     */
    private fun isSymmetricInternal(
      matrix: RealMatrix,
      relativeTolerance: Double,
      raiseException: Boolean
    ): Boolean {
      val rows: Int = matrix.rowDimension
      if (rows != matrix.columnDimension) {
        if (raiseException) {
          throw NonSquareMatrixException(rows, matrix.columnDimension)
        } else {
          return false
        }
      }
      for (i in 0..<rows) {
        for (j in i + 1..<rows) {
          val mij: Double = matrix[i][j]
          val mji: Double = matrix[j][i]
          if (abs(mij - mji) >
            max(abs(mij), abs(mji))*relativeTolerance
          ) {
            if (raiseException) {
              throw NonSymmetricMatrixException(i, j, relativeTolerance)
            } else {
              return false
            }
          }
        }
      }
      return true
    }
  }
}

class NonSquareMatrixException(val rowDimension: Int, val columnDimension: Int):
  RuntimeException("Matrix not square $rowDimension != $columnDimension")

class NonSymmetricMatrixException(val rowDimension: Int, val columnDimension: Int, threshold: Double):
  RuntimeException("Matrix not square $rowDimension != $columnDimension")

class MathUnsupportedOperationException(): RuntimeException()

class MathArithmeticException : RuntimeException()

class MaxCountExceededException(counter: Int) : RuntimeException()
