package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Class transforming a general real matrix to Schur form.
 * <p>A m &times; m matrix A can be written as the product of three matrices: A = P
 * &times; T &times; P<sup>T</sup> with P an orthogonal matrix and T an quasi-triangular
 * matrix. Both P and T are m &times; m matrices.</p>
 * <p>Transformation to Schur form is often not a goal by itself, but it is an
 * intermediate step in more general decomposition algorithms like
 * {@link EigenDecomposition eigen decomposition}. This class is therefore
 * intended for internal use by the library and is not public. As a consequence
 * of this explicitly limited scope, many methods directly returns references to
 * internal arrays, not copies.</p>
 * <p>This class is based on the method hqr2 in class EigenvalueDecomposition
 * from the <a href="http://math.nist.gov/javanumerics/jama/">JAMA</a> library.</p>
 *
 * @see <a href="http://mathworld.wolfram.com/SchurDecomposition.html">Schur Decomposition - MathWorld</a>
 * @see <a href="http://en.wikipedia.org/wiki/Schur_decomposition">Schur Decomposition - Wikipedia</a>
 * @see <a href="http://en.wikipedia.org/wiki/Householder_transformation">Householder Transformations</a>
 * @since 3.1
 */
class SchurTransformer(matrix: RealMatrix) {

  companion object {
    /** Maximum allowed iterations for convergence of the transformation. */
    private const val MAX_ITERATIONS = 100
  }

  /** P matrix. */
  private val matrixP: Array<DoubleArray>
  /** T matrix. */
  private val matrixT: Array<DoubleArray>
  /** Cached value of P. */
  private var cachedP: RealMatrix? = null
  /** Cached value of T. */
  private var cachedT: RealMatrix? = null
  /** Cached value of PT. */
  private var cachedPt: RealMatrix? = null
  /** Epsilon criteria taken from JAMA code (originally was 2^-52). */
  private val epsilon = Precision.EPSILON
  /**
   * Build the transformation to Schur form of a general real matrix.
   *
   * @param matrix matrix to transform
   * @throws NonSquareMatrixException if the matrix is not square
   */
  init {
    if (!matrix.isSquare) {
      throw NonSquareMatrixException(matrix.rowDimension, matrix.columnDimension)
    }
    val transformer = HessenbergTransformer(matrix)
    matrixT = transformer.h.getData()
    matrixP = transformer.p.getData()
    cachedT = null
    cachedP = null
    cachedPt = null
    transform()
  }
  /**
   * Returns the matrix P of the transform.
   * <p>P is an orthogonal matrix, i.e. its inverse is also its transpose.</p>
   *
   * @return the P matrix
   */
  val p: RealMatrix
    get() {
      if (cachedP == null) {
        cachedP = mk.ndarray(matrixP)
      }
      return cachedP!!
    }
  /**
   * Returns the transpose of the matrix P of the transform.
   * <p>P is an orthogonal matrix, i.e. its inverse is also its transpose.</p>
   *
   * @return the transpose of the P matrix
   */
  val pT: RealMatrix
    get() {
      if (cachedPt == null) {
        cachedPt = p.transpose()
      }
      return cachedPt!!
    }
  /**
   * Returns the quasi-triangular Schur matrix T of the transform.
   *
   * @return the T matrix
   */
  val t: RealMatrix
    get() {
      if (cachedT == null) {
        cachedT = mk.ndarray(matrixT)
      }
      return cachedT!!
    }
  /**
   * Transform original matrix to Schur form.
   * @throws MaxCountExceededException if the transformation does not converge
   */
  private fun transform() {
    val n = matrixT.size
    // compute matrix norm
    val norm = getNorm()
    // shift information
    val shift = ShiftInfo()
    // Outer loop over eigenvalue index
    var iteration = 0
    var iu = n - 1
    while (iu >= 0) {
      // Look for single small sub-diagonal element
      val il = findSmallSubDiagonalElement(iu, norm)
      // Check for convergence
      when {
        il == iu -> {
          // One root found
          matrixT[iu][iu] += shift.exShift
          iu--
          iteration = 0
        }

        il == iu - 1 -> {
          // Two roots found
          var p = (matrixT[iu - 1][iu - 1] - matrixT[iu][iu])/2.0
          var q = p*p + matrixT[iu][iu - 1]*matrixT[iu - 1][iu]
          matrixT[iu][iu] += shift.exShift
          matrixT[iu - 1][iu - 1] += shift.exShift
          if (q >= 0) {
            var z = sqrt(abs(q))
            z = if (p >= 0) p + z else p - z
            val x = matrixT[iu][iu - 1]
            val s = abs(x) + abs(z)
            p = x/s
            q = z/s
            val r = sqrt(p*p + q*q)
            p /= r
            q /= r
            // Row modification
            for (j in iu - 1 until n) {
              z = matrixT[iu - 1][j]
              matrixT[iu - 1][j] = q*z + p*matrixT[iu][j]
              matrixT[iu][j] = q*matrixT[iu][j] - p*z
            }
            // Column modification
            for (i in 0..iu) {
              z = matrixT[i][iu - 1]
              matrixT[i][iu - 1] = q*z + p*matrixT[i][iu]
              matrixT[i][iu] = q*matrixT[i][iu] - p*z
            }
            // Accumulate transformations
            for (i in 0 until n) {
              z = matrixP[i][iu - 1]
              matrixP[i][iu - 1] = q*z + p*matrixP[i][iu]
              matrixP[i][iu] = q*matrixP[i][iu] - p*z
            }
          }
          iu -= 2
          iteration = 0
        }
        else -> {
          // No convergence yet
          computeShift(il, iu, iteration, shift)
          // stop transformation after too many iterations
          if (++iteration > MAX_ITERATIONS) {
            throw MaxCountExceededException(MAX_ITERATIONS)
          }
          // the initial houseHolder vector for the QR step
          val hVec = DoubleArray(3)
          val im = initQRStep(il, iu, shift, hVec)
          performDoubleQRStep(il, im, iu, shift, hVec)
        }
      }
    }
  }
  /**
   * Computes the L1 norm of the (quasi-)triangular matrix T.
   *
   * @return the L1 norm of matrix T
   */
  private fun getNorm(): Double {
    var norm = 0.0
    for (i in matrixT.indices) {
      // as matrix T is (quasi-)triangular, also take the sub-diagonal element into account
      for (j in max(i - 1, 0) until matrixT.size) {
        norm += abs(matrixT[i][j])
      }
    }
    return norm
  }
  /**
   * Find the first small sub-diagonal element and returns its index.
   *
   * @param startIdx the starting index for the search
   * @param norm the L1 norm of the matrix
   * @return the index of the first small sub-diagonal element
   */
  private fun findSmallSubDiagonalElement(startIdx: Int, norm: Double): Int {
    var l = startIdx
    while (l > 0) {
      var s = abs(matrixT[l - 1][l - 1]) + abs(matrixT[l][l])
      if (s == 0.0) {
        s = norm
      }
      if (abs(matrixT[l][l - 1]) < epsilon * s) {
        break
      }
      l--
    }
    return l
  }
  /**
   * Compute the shift for the current iteration.
   *
   * @param l the index of the small sub-diagonal element
   * @param idx the current eigenvalue index
   * @param iteration the current iteration
   * @param shift holder for shift information
   */
  private fun computeShift(l: Int, idx: Int, iteration: Int, shift: ShiftInfo) {
    // Form shift
    shift.x = matrixT[idx][idx]
    shift.y = 0.0
    shift.w = 0.0
    if (l < idx) {
      shift.y = matrixT[idx - 1][idx - 1]
      shift.w = matrixT[idx][idx - 1] * matrixT[idx - 1][idx]
    }
    // Wilkinson's original ad hoc shift
    if (iteration == 10) {
      shift.exShift += shift.x
      for (i in 0..idx) {
        matrixT[i][i] -= shift.x
      }
      val s = abs(matrixT[idx][idx - 1]) + abs(matrixT[idx - 1][idx - 2])
      shift.x = 0.75 * s
      shift.y = 0.75 * s
      shift.w = -0.4375 * s * s
    }
    // MATLAB's new ad hoc shift
    if (iteration == 30) {
      var s = (shift.y - shift.x) / 2.0
      s = s * s + shift.w
      if (s > 0.0) {
        s = sqrt(s)
        if (shift.y < shift.x) {
          s = -s
        }
        s = shift.x - shift.w / ((shift.y - shift.x) / 2.0 + s)
        for (i in 0..idx) {
          matrixT[i][i] -= s
        }
        shift.exShift += s
        shift.x = 0.0
        shift.y = 0.0
        shift.w = 0.964
      }
    }
  }
  /**
   * Initialize the householder vectors for the QR step.
   *
   * @param il the index of the small sub-diagonal element
   * @param iu the current eigenvalue index
   * @param shift shift information holder
   * @param hVec the initial houseHolder vector
   * @return the start index for the QR step
   */
  private fun initQRStep(il: Int, iu: Int, shift: ShiftInfo, hVec: DoubleArray): Int {
    // Look for two consecutive small sub-diagonal elements
    var im = iu - 2
    while (im >= il) {
      val z = matrixT[im][im]
      val r = shift.x - z
      val s = shift.y - z
      hVec[0] = (r * s - shift.w) / matrixT[im + 1][im] + matrixT[im][im + 1]
      hVec[1] = matrixT[im + 1][im + 1] - z - r - s
      hVec[2] = matrixT[im + 2][im + 1]
      if (im == il) {
        break
      }
      val lhs = abs(matrixT[im][im - 1]) * (abs(hVec[1]) + abs(hVec[2]))
      val rhs = abs(hVec[0]) * (abs(matrixT[im - 1][im - 1]) +
        abs(z) +
        abs(matrixT[im + 1][im + 1]))
      if (lhs < epsilon * rhs) {
        break
      }
      im--
    }
    return im
  }
  /**
   * Perform a double QR step involving rows l:idx and columns m:n
   *
   * @param il the index of the small sub-diagonal element
   * @param im the start index for the QR step
   * @param iu the current eigenvalue index
   * @param shift shift information holder
   * @param hVec the initial houseHolder vector
   */
  private fun performDoubleQRStep(il: Int, im: Int, iu: Int, shift: ShiftInfo, hVec: DoubleArray) {
    val n = matrixT.size
    var p = hVec[0]
    var q = hVec[1]
    var r = hVec[2]
    for (k in im until iu) {
      val notLast = k != (iu - 1)
      if (k != im) {
        p = matrixT[k][k - 1]
        q = matrixT[k + 1][k - 1]
        r = if (notLast) matrixT[k + 2][k - 1] else 0.0
        shift.x = abs(p) + abs(q) + abs(r)
        if (Precision.equals(shift.x, 0.0, epsilon)) {
          continue
        }
        p /= shift.x
        q /= shift.x
        r /= shift.x
      }
      var s = sqrt(p * p + q * q + r * r)
      if (p < 0.0) {
        s = -s
      }
      if (s != 0.0) {
        if (k != im) {
          matrixT[k][k - 1] = -s * shift.x
        } else if (il != im) {
          matrixT[k][k - 1] = -matrixT[k][k - 1]
        }
        p += s
        shift.x = p / s
        shift.y = q / s
        val z = r / s
        q /= p
        r /= p
        // Row modification
        for (j in k until n) {
          var tempP = matrixT[k][j] + q * matrixT[k + 1][j]
          if (notLast) {
            tempP += r * matrixT[k + 2][j]
            matrixT[k + 2][j] -= tempP * z
          }
          matrixT[k][j] -= tempP * shift.x
          matrixT[k + 1][j] -= tempP * shift.y
        }
        // Column modification
        for (i in 0..minOf(iu, k + 3)) {
          var tempP = shift.x * matrixT[i][k] + shift.y * matrixT[i][k + 1]
          if (notLast) {
            tempP += z * matrixT[i][k + 2]
            matrixT[i][k + 2] -= tempP * r
          }
          matrixT[i][k] -= tempP
          matrixT[i][k + 1] -= tempP * q
        }
        // Accumulate transformations
        for (i in 0 until n) {
          var tempP = shift.x * matrixP[i][k] + shift.y * matrixP[i][k + 1]
          if (notLast) {
            tempP += z * matrixP[i][k + 2]
            matrixP[i][k + 2] -= tempP * r
          }
          matrixP[i][k] -= tempP
          matrixP[i][k + 1] -= tempP * q
        }
      }  // (s != 0)
    }  // k loop
    // clean up pollution due to round-off errors
    for (i in im + 2..iu) {
      matrixT[i][i - 2] = 0.0
      if (i > im + 2) {
        matrixT[i][i - 3] = 0.0
      }
    }
  }
  /**
   * Internal data structure holding the current shift information.
   * Contains variable names as present in the original JAMA code.
   */
  private data class ShiftInfo(
    // CHECKSTYLE: stop all
    /** x shift info */
    var x: Double = 0.0,
    /** y shift info */
    var y: Double = 0.0,
    /** w shift info */
    var w: Double = 0.0,
    /** Indicates an exceptional shift. */
    var exShift: Double = 0.0
    // CHECKSTYLE: resume all
  )
}