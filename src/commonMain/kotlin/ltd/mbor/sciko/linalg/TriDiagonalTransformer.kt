package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import java.util.*


class TriDiagonalTransformer {
  /** Householder vectors.  */
  private var householderVectors: Array<DoubleArray>

  /** Main diagonal.  */
  private var main: DoubleArray

  /** Secondary diagonal.  */
  private var secondary: DoubleArray

  /** Cached value of Q.  */
  private var cachedQ: RealMatrix? = null

  /** Cached value of Qt.  */
  private var cachedQt: RealMatrix? = null

  /** Cached value of T.  */
  private var cachedT: RealMatrix? = null

  /**
   * Build the transformation to tridiagonal shape of a symmetrical matrix.
   *
   * The specified matrix is assumed to be symmetrical without any check.
   * Only the upper triangular part of the matrix is used.
   *
   * @param matrix Symmetrical matrix to transform.
   * @throws NonSquareMatrixException if the matrix is not square.
   */
  constructor(matrix: RealMatrix) {
    if (!matrix.isSquare) {
      throw NonSquareMatrixException(
        matrix.rowDimension,
        matrix.columnDimension
      )
    }
    val m: Int = matrix.rowDimension
    householderVectors = matrix.toArray()
    main = DoubleArray(m)
    secondary = DoubleArray(m - 1)
    cachedQ = null
    cachedQt = null
    cachedT = null
    // transform matrix
    transform()
  }

  /**
   * Returns the matrix Q of the transform.
   *
   * Q is an orthogonal matrix, i.e. its transpose is also its inverse.
   * @return the Q matrix
   */
  val q: RealMatrix? get() {
    if (cachedQ == null) {
      cachedQ = getQT()!!.transpose()
    }
    return cachedQ
  }

  /**
   * Returns the transpose of the matrix Q of the transform.
   *
   * Q is an orthogonal matrix, i.e. its transpose is also its inverse.
   * @return the Q matrix
   */
  fun getQT(): RealMatrix? {
    if (cachedQt == null) {
      val m = householderVectors.size
      val qta = Array<DoubleArray>(m) { DoubleArray(m) }
      // build up first part of the matrix by applying Householder transforms
      for (k in m - 1 downTo 1) {
        val hK: DoubleArray = householderVectors[k - 1]
        qta[k][k] = 1.0
        if (hK[k] != 0.0) {
          val inv = 1.0/(secondary[k - 1]*hK[k])
          var beta = 1.0/secondary[k - 1]
          qta[k][k] = 1 + beta*hK[k]
          for (i in k + 1..<m) {
            qta[k][i] = beta*hK[i]
          }
          for (j in k + 1..<m) {
            beta = 0.0
            for (i in k + 1..<m) {
              beta += qta[j][i]*hK[i]
            }
            beta *= inv
            qta[j][k] = beta*hK[k]
            for (i in k + 1..<m) {
              qta[j][i] += beta*hK[i]
            }
          }
        }
      }
      qta[0][0] = 1.0
      cachedQt = mk.ndarray(qta)
    }
    // return the cached matrix
    return cachedQt
  }

  /**
   * Returns the tridiagonal matrix T of the transform.
   * @return the T matrix
   */
  fun getT(): RealMatrix? {
    if (cachedT == null) {
      val m = main.size
      val ta = Array(m) { DoubleArray(m) }
      for (i in 0..<m) {
        ta[i][i] = main[i]
        if (i > 0) {
          ta[i][i - 1] = secondary[i - 1]
        }
        if (i < (main.size - 1)) {
          ta[i]!![i + 1] = secondary[i]
        }
      }
      cachedT = mk.ndarray(ta)
    }
    // return the cached matrix
    return cachedT
  }

  /**
   * Get the Householder vectors of the transform.
   *
   * Note that since this class is only intended for internal use,
   * it returns directly a reference to its internal arrays, not a copy.
   * @return the main diagonal elements of the B matrix
   */
  fun getHouseholderVectorsRef(): Array<DoubleArray> {
    return householderVectors
  }

  /**
   * Get the main diagonal elements of the matrix T of the transform.
   *
   * Note that since this class is only intended for internal use,
   * it returns directly a reference to its internal arrays, not a copy.
   * @return the main diagonal elements of the T matrix
   */
  val mainDiagonalRef: DoubleArray get() {
    return main
  }

  /**
   * Get the secondary diagonal elements of the matrix T of the transform.
   *
   * Note that since this class is only intended for internal use,
   * it returns directly a reference to its internal arrays, not a copy.
   * @return the secondary diagonal elements of the T matrix
   */
  val secondaryDiagonalRef: DoubleArray get() {
    return secondary
  }

  /**
   * Transform original matrix to tridiagonal form.
   *
   * Transformation is done using Householder transforms.
   */
  private fun transform() {
    val m = householderVectors.size
    val z = DoubleArray(m)
    for (k in 0..<m - 1) {
      //zero-out a row and a column simultaneously
      val hK: DoubleArray = householderVectors[k]
      main[k] = hK[k]
      var xNormSqr = 0.0
      for (j in k + 1..<m) {
        val c = hK[j]
        xNormSqr += c*c
      }
      val a = if (hK[k + 1] > 0) -FastMath.sqrt(xNormSqr) else FastMath.sqrt(xNormSqr)
      secondary[k] = a
      if (a != 0.0) {
        // apply Householder transform from left and right simultaneously
        hK[k + 1] -= a
        val beta = -1/(a*hK[k + 1])
        // compute a = beta A v, where v is the Householder vector
        // this loop is written in such a way
        //   1) only the upper triangular part of the matrix is accessed
        //   2) access is cache-friendly for a matrix stored in rows
        Arrays.fill(z, k + 1, m, 0.0)
        for (i in k + 1..<m) {
          val hI: DoubleArray = householderVectors[i]
          val hKI = hK[i]
          var zI = hI[i]*hKI
          for (j in i + 1..<m) {
            val hIJ = hI[j]
            zI += hIJ*hK[j]
            z[j] += hIJ*hKI
          }
          z[i] = beta*(z[i] + zI)
        }
        // compute gamma = beta vT z / 2
        var gamma = 0.0
        for (i in k + 1..<m) {
          gamma += z[i]*hK[i]
        }
        gamma *= beta/2
        // compute z = z - gamma v
        for (i in k + 1..<m) {
          z[i] -= gamma*hK[i]
        }
        // update matrix: A = A - v zT - z vT
        // only the upper triangular part of the matrix is updated
        for (i in k + 1..<m) {
          val hI: DoubleArray = householderVectors[i]
          for (j in i..<m) {
            hI[j] -= hK[i]*z[j] + z[i]*hK[j]
          }
        }
      }
    }
    main[m - 1] = householderVectors[m - 1][m - 1]
  }
}