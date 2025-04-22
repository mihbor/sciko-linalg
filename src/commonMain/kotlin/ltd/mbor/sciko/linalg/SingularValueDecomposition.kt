package ltd.mbor.sciko.linalg

import ltd.mbor.sciko.linalg.Precision.Companion.SAFE_MIN
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray


/**
 * Calculates the compact Singular Value Decomposition of a matrix.
 *
 *
 * The Singular Value Decomposition of matrix A is a set of three matrices: U,
 * &Sigma; and V such that A = U &times; &Sigma; &times; V<sup>T</sup>. Let A be
 * a m &times; n matrix, then U is a m &times; p orthogonal matrix, &Sigma; is a
 * p &times; p diagonal matrix with positive or null elements, V is a p &times;
 * n orthogonal matrix (hence V<sup>T</sup> is also orthogonal) where
 * p=min(m,n).
 *
 * This class is similar to the class with similar name from the
 * [JAMA](http://math.nist.gov/javanumerics/jama/) library, with the
 * following changes:
 *
 *  * the `norm2` method which has been renamed as [   getNorm][.getNorm],
 *  * the `cond` method which has been renamed as [   ][.getConditionNumber],
 *  * the `rank` method which has been renamed as [   getRank][.getRank],
 *  * a [getUT][.getUT] method has been added,
 *  * a [getVT][.getVT] method has been added,
 *  * a [getSolver][.getSolver] method has been added,
 *  * a [getCovariance][.getCovariance] method has been added.
 *
 * @see [MathWorld](http://mathworld.wolfram.com/SingularValueDecomposition.html)
 *
 * @see [Wikipedia](http://en.wikipedia.org/wiki/Singular_value_decomposition)
 *
 * @since 2.0 (changed to concrete class in 3.0)
 */
class SingularValueDecomposition(matrix: RealMatrix) {
  /** Computed singular values.  */
  internal val singularValues: DoubleArray

  /** max(row dimension, column dimension).  */
  private var m = 0

  /** min(row dimension, column dimension).  */
  private var n = 0

  /** Indicator for transposed matrix.  */
  private var transposed = false// return the cached matrix
  /**
   * Returns the matrix U of the decomposition.
   *
   * U is an orthogonal matrix, i.e. its transpose is also its inverse.
   * @return the U matrix
   * @see .getUT
   */
  /** Cached value of U matrix.  */
  var u: RealMatrix

  /** Cached value of transposed U matrix.  */
  var cachedUt: RealMatrix? = null

  /** Cached value of S (diagonal) matrix.  */
  var cachedS: RealMatrix? = null
  /**
   * Returns the matrix V of the decomposition.
   *
   * V is an orthogonal matrix, i.e. its transpose is also its inverse.
   * @return the V matrix (or null if decomposed matrix is singular)
   * @see .getVT
   */
  /** Cached value of V matrix.  */
  var v: RealMatrix

  /** Cached value of transposed V matrix.  */
  var cachedVt: RealMatrix? = null

  /**
   * Tolerance value for small singular values, calculated once we have
   * populated "singularValues".
   */
  private val tol: Double

  /**
   * Calculates the compact Singular Value Decomposition of the given matrix.
   *
   * @param matrix Matrix to decompose.
   */
  init {
    val A: Array<DoubleArray>
    // "m" is always the largest dimension.
    if (matrix.rowDimension < matrix.columnDimension) {
      transposed = true
      A = matrix.transpose().toArray()
      m = matrix.columnDimension
      n = matrix.rowDimension
    } else {
      transposed = false
      A = matrix.toArray()
      m = matrix.rowDimension
      n = matrix.columnDimension
    }
    singularValues = DoubleArray(n)
    val U = Array(m) { DoubleArray(n) }
    val V = Array(n) { DoubleArray(n) }
    val e = DoubleArray(n)
    val work = DoubleArray(m)
    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.
    val nct = FastMath.min(m - 1, n)
    val nrt = FastMath.max(0, n - 2)
    for (k in 0 until FastMath.max(nct, nrt)) {
      if (k < nct) {
        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.
        singularValues[k] = 0.0
        for (i in k until m) {
          singularValues[k] = FastMath.hypot(singularValues[k], A[i][k])
        }
        if (singularValues[k] != 0.0) {
          if (A[k][k] < 0) {
            singularValues[k] = -singularValues[k]
          }
          for (i in k until m) {
            A[i][k] /= singularValues[k]
          }
          A[k][k] += 1.0
        }
        singularValues[k] = -singularValues[k]
      }
      for (j in k + 1 until n) {
        if (k < nct &&
          singularValues[k] != 0.0
        ) {
          // Apply the transformation.
          var t = 0.0
          for (i in k until m) {
            t += A[i][k] * A[i][j]
          }
          t = -t / A[k][k]
          for (i in k until m) {
            A[i][j] += t * A[i][k]
          }
        }
        // Place the k-th row of A into e for the
        // subsequent calculation of the row transformation.
        e[j] = A[k][j]
      }
      if (k < nct) {
        // Place the transformation in U for subsequent back
        // multiplication.
        for (i in k until m) {
          U[i][k] = A[i][k]
        }
      }
      if (k < nrt) {
        // Compute the k-th row transformation and place the
        // k-th super-diagonal in e[k].
        // Compute 2-norm without under/overflow.
        e[k] = 0.0
        for (i in k + 1 until n) {
          e[k] = FastMath.hypot(e[k], e[i])
        }
        if (e[k] != 0.0) {
          if (e[k + 1] < 0) {
            e[k] = -e[k]
          }
          for (i in k + 1 until n) {
            e[i] /= e[k]
          }
          e[k + 1] += 1.0
        }
        e[k] = -e[k]
        if (k + 1 < m &&
          e[k] != 0.0
        ) {
          // Apply the transformation.
          for (i in k + 1 until m) {
            work[i] = 0.0
          }
          for (j in k + 1 until n) {
            for (i in k + 1 until m) {
              work[i] += e[j] * A[i][j]
            }
          }
          for (j in k + 1 until n) {
            val t = -e[j] / e[k + 1]
            for (i in k + 1 until m) {
              A[i][j] += t * work[i]
            }
          }
        }
        // Place the transformation in V for subsequent
        // back multiplication.
        for (i in k + 1 until n) {
          V[i][k] = e[i]
        }
      }
    }
    // Set up the final bidiagonal matrix or order p.
    var p = n
    if (nct < n) {
      singularValues[nct] = A[nct][nct]
    }
    if (m < p) {
      singularValues[p - 1] = 0.0
    }
    if (nrt + 1 < p) {
      e[nrt] = A[nrt][p - 1]
    }
    e[p - 1] = 0.0
    // Generate U.
    for (j in nct until n) {
      for (i in 0 until m) {
        U[i][j] = 0.0
      }
      U[j][j] = 1.0
    }
    for (k in nct - 1 downTo 0) {
      if (singularValues[k] != 0.0) {
        for (j in k + 1 until n) {
          var t = 0.0
          for (i in k until m) {
            t += U[i][k] * U[i][j]
          }
          t = -t / U[k][k]
          for (i in k until m) {
            U[i][j] += t * U[i][k]
          }
        }
        for (i in k until m) {
          U[i][k] = -U[i][k]
        }
        U[k][k] = 1 + U[k][k]
        for (i in 0 until k - 1) {
          U[i][k] = 0.0
        }
      } else {
        for (i in 0 until m) {
          U[i][k] = 0.0
        }
        U[k][k] = 1.0
      }
    }
    // Generate V.
    for (k in n - 1 downTo 0) {
      if (k < nrt &&
        e[k] != 0.0
      ) {
        for (j in k + 1 until n) {
          var t = 0.0
          for (i in k + 1 until n) {
            t += V[i][k] * V[i][j]
          }
          t = -t / V[k + 1][k]
          for (i in k + 1 until n) {
            V[i][j] += t * V[i][k]
          }
        }
      }
      for (i in 0 until n) {
        V[i][k] = 0.0
      }
      V[k][k] = 1.0
    }
    // Main iteration loop for the singular values.
    val pp = p - 1
    while (p > 0) {
      var k: Int
      var kase: Int
      // Here is where a test for too many iterations would go.
      // This section of the program inspects for
      // negligible elements in the s and e arrays.  On
      // completion the variables kase and k are set as follows.
      // kase = 1     if s(p) and e[k-1] are negligible and k<p
      // kase = 2     if s(k) is negligible and k<p
      // kase = 3     if e[k-1] is negligible, k<p, and
      //              s(k), ..., s(p) are not negligible (qr step).
      // kase = 4     if e(p-1) is negligible (convergence).
      k = p - 2
      while (k >= 0) {
        val threshold = TINY + EPS * (FastMath.abs(
          singularValues[k]
        ) +
          FastMath.abs(singularValues[k + 1]))
        // the following condition is written this way in order
        // to break out of the loop when NaN occurs, writing it
        // as "if (FastMath.abs(e[k]) <= threshold)" would loop
        // indefinitely in case of NaNs because comparison on NaNs
        // always return false, regardless of what is checked
        // see issue MATH-947
        if (!(FastMath.abs(e[k]) > threshold)) {
          e[k] = 0.0
          break
        }
        k--
      }
      if (k == p - 2) {
        kase = 4
      } else {
        var ks = p - 1
        while (ks >= k) {
          if (ks == k) {
            break
          }
          val t = (if (ks != p) FastMath.abs(e[ks]) else 0.0) +
            (if (ks != k + 1) FastMath.abs(e[ks - 1]) else 0.0)
          if (FastMath.abs(singularValues[ks]) <= TINY + EPS * t) {
            singularValues[ks] = 0.0
            break
          }
          ks--
        }
        if (ks == k) {
          kase = 3
        } else if (ks == p - 1) {
          kase = 1
        } else {
          kase = 2
          k = ks
        }
      }
      k++
      // Perform the task indicated by kase.
      when (kase) {
        1 -> {
          var f = e[p - 2]
          e[p - 2] = 0.0
          var j = p - 2
          while (j >= k) {
            var t = FastMath.hypot(singularValues[j], f)
            val cs = singularValues[j] / t
            val sn = f / t
            singularValues[j] = t
            if (j != k) {
              f = -sn * e[j - 1]
              e[j - 1] = cs * e[j - 1]
            }
            var i = 0
            while (i < n) {
              t = cs * V[i][j] + sn * V[i][p - 1]
              V[i][p - 1] = -sn * V[i][j] + cs * V[i][p - 1]
              V[i][j] = t
              i++
            }
            j--
          }
        }

        2 -> {
          var f = e[k - 1]
          e[k - 1] = 0.0
          var j = k
          while (j < p) {
            var t = FastMath.hypot(singularValues[j], f)
            val cs = singularValues[j] / t
            val sn = f / t
            singularValues[j] = t
            f = -sn * e[j]
            e[j] = cs * e[j]
            var i = 0
            while (i < m) {
              t = cs * U[i][j] + sn * U[i][k - 1]
              U[i][k - 1] = -sn * U[i][j] + cs * U[i][k - 1]
              U[i][j] = t
              i++
            }
            j++
          }
        }

        3 -> {
          // Calculate the shift.
          val maxPm1Pm2 = FastMath.max(
            FastMath.abs(singularValues[p - 1]),
            FastMath.abs(singularValues[p - 2])
          )
          val scale = FastMath.max(
            FastMath.max(
              FastMath.max(
                maxPm1Pm2,
                FastMath.abs(e[p - 2])
              ),
              FastMath.abs(singularValues[k])
            ),
            FastMath.abs(e[k])
          )
          val sp = singularValues[p - 1] / scale
          val spm1 = singularValues[p - 2] / scale
          val epm1 = e[p - 2] / scale
          val sk = singularValues[k] / scale
          val ek = e[k] / scale
          val b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0
          val c = (sp * epm1) * (sp * epm1)
          var shift = 0.0
          if (b != 0.0 ||
            c != 0.0
          ) {
            shift = FastMath.sqrt(b * b + c)
            if (b < 0) {
              shift = -shift
            }
            shift = c / (b + shift)
          }
          var f = (sk + sp) * (sk - sp) + shift
          var g = sk * ek
          // Chase zeros.
          var j = k
          while (j < p - 1) {
            var t = FastMath.hypot(f, g)
            var cs = f / t
            var sn = g / t
            if (j != k) {
              e[j - 1] = t
            }
            f = cs * singularValues[j] + sn * e[j]
            e[j] = cs * e[j] - sn * singularValues[j]
            g = sn * singularValues[j + 1]
            singularValues[j + 1] = cs * singularValues[j + 1]
            var i = 0
            while (i < n) {
              t = cs * V[i][j] + sn * V[i][j + 1]
              V[i][j + 1] = -sn * V[i][j] + cs * V[i][j + 1]
              V[i][j] = t
              i++
            }
            t = FastMath.hypot(f, g)
            cs = f / t
            sn = g / t
            singularValues[j] = t
            f = cs * e[j] + sn * singularValues[j + 1]
            singularValues[j + 1] = -sn * e[j] + cs * singularValues[j + 1]
            g = sn * e[j + 1]
            e[j + 1] = cs * e[j + 1]
            if (j < m - 1) {
              var i = 0
              while (i < m) {
                t = cs * U[i][j] + sn * U[i][j + 1]
                U[i][j + 1] = -sn * U[i][j] + cs * U[i][j + 1]
                U[i][j] = t
                i++
              }
            }
            j++
          }
          e[p - 2] = f
        }

        else -> {
          // Make the singular values positive.
          if (singularValues[k] <= 0) {
            singularValues[k] = if (singularValues[k] < 0) -singularValues[k] else 0.0
            var i = 0
            while (i <= pp) {
              V[i][k] = -V[i][k]
              i++
            }
          }
          // Order the singular values.
          while (k < pp) {
            if (singularValues[k] >= singularValues[k + 1]) {
              break
            }
            var t = singularValues[k]
            singularValues[k] = singularValues[k + 1]
            singularValues[k + 1] = t
            if (k < n - 1) {
              var i = 0
              while (i < n) {
                t = V[i][k + 1]
                V[i][k + 1] = V[i][k]
                V[i][k] = t
                i++
              }
            }
            if (k < m - 1) {
              var i = 0
              while (i < m) {
                t = U[i][k + 1]
                U[i][k + 1] = U[i][k]
                U[i][k] = t
                i++
              }
            }
            k++
          }
          p--
        }
      }
    }
    // Set the small value tolerance used to calculate rank and pseudo-inverse
    tol = FastMath.max(
      m * singularValues[0] * EPS,
      FastMath.sqrt(SAFE_MIN)
    )
    if (!transposed) {
      u = mk.ndarray(U)
      v = mk.ndarray(V)
    } else {
      u = mk.ndarray(V)
      v = mk.ndarray(U)
    }
  }

  val uT: RealMatrix?
    /**
     * Returns the transpose of the matrix U of the decomposition.
     *
     * U is an orthogonal matrix, i.e. its transpose is also its inverse.
     * @return the U matrix (or null if decomposed matrix is singular)
     * @see .getU
     */
    get() {
      if (cachedUt == null) {
        cachedUt = u.transpose()
      }
      // return the cached matrix
      return cachedUt
    }
  val s: RealMatrix
    /**
     * Returns the diagonal matrix  of the decomposition.
     *
     *  is a diagonal matrix. The singular values are provided in
     * non-increasing order, for compatibility with Jama.
     * @return the  matrix
     */
    get() {
      if (cachedS == null) {
        // cache the matrix for subsequent calls
        cachedS = mk.diagonal(singularValues.toList())
      }
      return cachedS!!
    }

  /**
   * Returns the diagonal elements of the matrix  of the decomposition.
   *
   * The singular values are provided in non-increasing order, for
   * compatibility with Jama.
   * @return the diagonal elements of the  matrix
   */
  fun getSingularValues(): DoubleArray {
    return singularValues.clone()
  }

  val vT: RealMatrix
    /**
     * Returns the transpose of the matrix V of the decomposition.
     *
     * V is an orthogonal matrix, i.e. its transpose is also its inverse.
     * @return the V matrix (or null if decomposed matrix is singular)
     * @see .getV
     */
    get() {
      return cachedVt ?: v.transpose().also{
        cachedVt = it
      }
    }

  /**
   * Returns the n  n covariance matrix.
   *
   * The covariance matrix is V  J  V<sup>T</sup>
   * where J is the diagonal matrix of the inverse of the squares of
   * the singular values.
   * @param minSingularValue value below which singular values are ignored
   * (a 0 or negative value implies all singular value will be used)
   * @return covariance matrix
   * @exception IllegalArgumentException if minSingularValue is larger than
   * the largest singular value, meaning all singular values are ignored
   */
  fun getCovariance(minSingularValue: Double): RealMatrix {
    // get the number of singular values to consider
    val p = singularValues.size
    var dimension = 0
    while (dimension < p &&
      singularValues[dimension] >= minSingularValue
    ) {
      ++dimension
    }
    if (dimension == 0) {
      throw NumberIsTooLargeException(
        minSingularValue, singularValues[0], true
      )
    }
    val data = Array(dimension) { DoubleArray(p) }
    vT!!.walkInOptimizedOrder(object : DefaultRealMatrixPreservingVisitor() {
      /** {@inheritDoc}  */
      override fun visit(
        row: Int, column: Int,
        value: Double
      ) {
        data[row][column] = value / singularValues[row]
      }
    }, 0, dimension - 1, 0, p - 1)
    val jv: RealMatrix = mk.ndarray(data)
    return jv.transpose() dot jv
  }

  val norm: Double
    /**
     * Returns the L<sub>2</sub> norm of the matrix.
     *
     * The L<sub>2</sub> norm is max(|A  u|<sub>2</sub> /
     * |u|<sub>2</sub>), where |.|<sub>2</sub> denotes the vectorial 2-norm
     * (i.e. the traditional euclidian norm).
     * @return norm
     */
    get() = singularValues[0]
  val conditionNumber: Double
    /**
     * Return the condition number of the matrix.
     * @return condition number of the matrix
     */
    get() = singularValues[0] / singularValues[n - 1]
  val inverseConditionNumber: Double
    /**
     * Computes the inverse of the condition number.
     * In cases of rank deficiency, the [condition][.getConditionNumber] will become undefined.
     *
     * @return the inverse of the condition number.
     */
    get() = singularValues[n - 1] / singularValues[0]
  val rank: Int
    /**
     * Return the effective numerical matrix rank.
     *
     * The effective numerical rank is the number of non-negligible
     * singular values. The threshold used to identify non-negligible
     * terms is max(m,n)  ulp(s<sub>1</sub>) where ulp(s<sub>1</sub>)
     * is the least significant bit of the largest singular value.
     * @return effective numerical matrix rank
     */
    get() {
      var r = 0
      for (i in singularValues.indices) {
        if (singularValues[i] > tol) {
          r++
        }
      }
      return r
    }
  val solver: Solver
    /**
     * Get a solver for finding the A  X = B solution in least square sense.
     * @return a solver
     */
    get() = Solver(singularValues, uT!!, v, rank == m, tol)

  /** Specialized solver.  */
  class Solver(
    singularValues: DoubleArray, uT: RealMatrix,
    v: RealMatrix, nonSingular: Boolean, tol: Double
  ) {
    /** Pseudo-inverse of the initial matrix.  */
    private val pseudoInverse: RealMatrix

    /** Singularity indicator.  */
    private val nonSingular: Boolean

    /**
     * Build a solver from decomposed matrix.
     *
     * @param singularValues Singular values.
     * @param uT U<sup>T</sup> matrix of the decomposition.
     * @param v V matrix of the decomposition.
     * @param nonSingular Singularity indicator.
     * @param tol tolerance for singular values
     */
    init {
      val suT = mk.ndarray(uT.toArray())
      for (i in singularValues.indices) {
        val a = if (singularValues[i] > tol) {
          1 / singularValues[i]
        } else {
          0.0
        }
        for (j in suT[i].indices) {
          suT[i, j] *= a
        }
      }
      pseudoInverse = v dot suT
      this.nonSingular = nonSingular
    }

    /**
     * Solve the linear equation A  X = B in least square sense.
     *
     *
     * The mn matrix A may not be square, the solution X is such that
     * ||A  X - B|| is minimal.
     *
     * @param b Right-hand side of the equation A  X = B
     * @return a vector X that minimizes the two norm of A  X - B
     * @throws org.apache.commons.math3.exception.DimensionMismatchException
     * if the matrices dimensions do not match.
     */
    @JvmName("solveRealVector")
    fun solve(b: RealVector): RealVector {
      return pseudoInverse dot b
    }

    /**
     * Solve the linear equation A  X = B in least square sense.
     *
     *
     * The mn matrix A may not be square, the solution X is such that
     * ||A  X - B|| is minimal.
     *
     *
     * @param b Right-hand side of the equation A  X = B
     * @return a matrix X that minimizes the two norm of A  X - B
     * @throws org.apache.commons.math3.exception.DimensionMismatchException
     * if the matrices dimensions do not match.
     */
    @JvmName("solveRealMatrix")
    fun solve(b: RealMatrix): RealMatrix {
      return pseudoInverse dot b
    }

    /**
     * Check if the decomposed matrix is non-singular.
     *
     * @return `true` if the decomposed matrix is non-singular.
     */
    val isNonSingular: Boolean
      get() {
        return nonSingular
      }

    /**
     * Get the pseudo-inverse of the decomposed matrix.
     *
     * @return the inverse matrix.
     */
    val inverse: RealMatrix
      get() {
        return pseudoInverse
      }
  }

  companion object {
    /** Relative threshold for small singular values.  */
    private const val EPS = 2.220446049250313E-16

    /** Absolute threshold for small singular values.  */
    private const val TINY = 1.6033346880071782E-291
  }
}

class NumberIsTooLargeException(minSingularValue: Double, d: Double, b: Boolean) : Throwable()

