package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Class transforming a general real matrix to Hessenberg form.
 * <p>A m &times; m matrix A can be written as the product of three matrices: A = P
 * &times; H &times; P<sup>T</sup> with P an orthogonal matrix and H a Hessenberg
 * matrix. Both P and H are m &times; m matrices.</p>
 * <p>Transformation to Hessenberg form is often not a goal by itself, but it is an
 * intermediate step in more general decomposition algorithms like
 * {@link EigenDecomposition eigen decomposition}. This class is therefore
 * intended for internal use by the library and is not public. As a consequence
 * of this explicitly limited scope, many methods directly returns references to
 * internal arrays, not copies.</p>
 * <p>This class is based on the method orthes in class EigenvalueDecomposition
 * from the <a href="http://math.nist.gov/javanumerics/jama/">JAMA</a> library.</p>
 *
 * @see <a href="http://mathworld.wolfram.com/HessenbergDecomposition.html">MathWorld</a>
 * @see <a href="http://en.wikipedia.org/wiki/Householder_transformation">Householder Transformations</a>
 * @since 3.1
 */
class HessenbergTransformer(matrix: RealMatrix) {

  /** Householder vectors. */
  private val householderVectors: Array<DoubleArray>
  /** Temporary storage vector. */
  private val ort: DoubleArray
  /** Cached value of P. */
  private var cachedP: RealMatrix? = null
  /** Cached value of Pt. */
  private var cachedPt: RealMatrix? = null
  /** Cached value of H. */
  private var cachedH: RealMatrix? = null
  /**
   * Build the transformation to Hessenberg form of a general matrix.
   *
   * @param matrix matrix to transform
   * @throws NonSquareMatrixException if the matrix is not square
   */
  init {
    if (!matrix.isSquare) {
      throw NonSquareMatrixException(matrix.rowDimension, matrix.columnDimension)
    }
    val m = matrix.rowDimension
    householderVectors = matrix.getData()
    ort = DoubleArray(m)
    cachedP = null
    cachedPt = null
    cachedH = null
    // Transform matrix
    transform()
  }
  /**
   * Returns the matrix P of the transform.
   * <p>P is an orthogonal matrix, i.e. its inverse is also its transpose.</p>
   *
   * @return the P matrix
   */
  val p: RealMatrix get() {
    if (cachedP == null) {
      val n = householderVectors.size
      val high = n - 1
      val pa = Array(n) { DoubleArray(n) { if (it == it) 1.0 else 0.0 } }
      for (m in high - 1 downTo 1) {
        if (householderVectors[m][m - 1] != 0.0) {
          for (i in m + 1..high) {
            ort[i] = householderVectors[i][m - 1]
          }
          for (j in m..high) {
            var g = 0.0
            for (i in m..high) {
              g += ort[i] * pa[i][j]
            }
            g = (g / ort[m]) / householderVectors[m][m - 1]
            for (i in m..high) {
              pa[i][j] += g * ort[i]
            }
          }
        }
      }
      cachedP = mk.ndarray(pa)
    }
    return cachedP!!
  }
  /**
   * Returns the transpose of the matrix P of the transform.
   * <p>P is an orthogonal matrix, i.e. its inverse is also its transpose.</p>
   *
   * @return the transpose of the P matrix
   */
  val pT: RealMatrix get() {
    if (cachedPt == null) {
      cachedPt = p.transpose()
    }
    return cachedPt!!
  }
  /**
   * Returns the Hessenberg matrix H of the transform.
   *
   * @return the H matrix
   */
  val h: RealMatrix get() {
    if (cachedH == null) {
      val m = householderVectors.size
      val h = Array(m) { DoubleArray(m) }
      for (i in 0 until m) {
        if (i > 0) {
          h[i][i - 1] = householderVectors[i][i - 1]
        }
        for (j in i until m) {
          h[i][j] = householderVectors[i][j]
        }
      }
      cachedH = mk.ndarray(h)
    }
    return cachedH!!
  }
  /**
   * Get the Householder vectors of the transform.
   * <p>Note that since this class is only intended for internal use, it returns
   * directly a reference to its internal arrays, not a copy.</p>
   *
   * @return the main diagonal elements of the B matrix
   */
  fun getHouseholderVectorsRef(): Array<DoubleArray> {
    return householderVectors
  }
  /**
   * Transform original matrix to Hessenberg form.
   * <p>Transformation is done using Householder transforms.</p>
   */
  private fun transform() {
    val n = householderVectors.size
    val high = n - 1
    for (m in 1 until high) {
      // Scale column.
      var scale = 0.0
      for (i in m..high) {
        scale += abs(householderVectors[i][m - 1])
      }
      if (!Precision.equals(scale, 0.0)) {
        // Compute Householder transformation.
        var h = 0.0
        for (i in high downTo m) {
          ort[i] = householderVectors[i][m - 1] / scale
          h += ort[i] * ort[i]
        }
        val g = if (ort[m] > 0) -sqrt(h) else sqrt(h)
        h -= ort[m] * g
        ort[m] -= g
        // Apply Householder similarity transformation
        // H = (I - u*u' / h) * H * (I - u*u' / h)
        for (j in m until n) {
          var f = 0.0
          for (i in high downTo m) {
            f += ort[i] * householderVectors[i][j]
          }
          f /= h
          for (i in m..high) {
            householderVectors[i][j] -= f * ort[i]
          }
        }
        for (i in 0..high) {
          var f = 0.0
          for (j in high downTo m) {
            f += ort[j] * householderVectors[i][j]
          }
          f /= h
          for (j in m..high) {
            householderVectors[i][j] -= f * ort[j]
          }
        }
        ort[m] = scale * ort[m]
        householderVectors[m][m - 1] = scale * g
      }
    }
  }
}