package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.Dim1
import org.jetbrains.kotlinx.multik.ndarray.data.Dim2
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.MutableMultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Calculates the eigen decomposition of a real matrix.
 * <p>The eigen decomposition of matrix A is a set of two matrices:
 * V and D such that A = V &times; D &times; V<sup>T</sup>.
 * A, V and D are all m &times; m matrices.</p>
 * <p>This class is similar in spirit to the <code>EigenvalueDecomposition</code>
 * class from the <a href="http://math.nist.gov/javanumerics/jama/">JAMA</a>
 * library, with the following changes:</p>
 * <ul>
 *   <li>a {@link #getVT() getVt} method has been added,</li>
 *   <li>two {@link #getRealEigenvalue(int) getRealEigenvalue} and {@link #getImagEigenvalue(int)
 *   getImagEigenvalue} methods to pick up a single eigenvalue have been added,</li>
 *   <li>a {@link #getEigenvector(int) getEigenvector} method to pick up a single
 *   eigenvector has been added,</li>
 *   <li>a {@link #getDeterminant() getDeterminant} method has been added.</li>
 *   <li>a {@link #getSolver() getSolver} method has been added.</li>
 * </ul>
 * <p>
 * As of 3.1, this class supports general real matrices (both symmetric and non-symmetric):
 * </p>
 * <p>
 * If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is diagonal and the eigenvector
 * matrix V is orthogonal, i.e. A = V.multiply(D.multiply(V.transpose())) and
 * V.multiply(V.transpose()) equals the identity matrix.
 * </p>
 * <p>
 * If A is not symmetric, then the eigenvalue matrix D is block diagonal with the real eigenvalues
 * in 1-by-1 blocks and any complex eigenvalues, lambda + i*mu, in 2-by-2 blocks:
 * <pre>
 *    [lambda, mu    ]
 *    [   -mu, lambda]
 * </pre>
 * The columns of V represent the eigenvectors in the sense that A*V = V*D,
 * i.e. A.multiply(V) equals V.multiply(D).
 * The matrix V may be badly conditioned, or even singular, so the validity of the equation
 * A = V*D*inverse(V) depends upon the condition of V.
 * </p>
 * <p>
 * This implementation is based on the paper by A. Drubrulle, R.S. Martin and
 * J.H. Wilkinson "The Implicit QL Algorithm" in Wilksinson and Reinsch (1971)
 * Handbook for automatic computation, vol. 2, Linear algebra, Springer-Verlag,
 * New-York
 * </p>
 * @see <a href="http://mathworld.wolfram.com/EigenDecomposition.html">MathWorld</a>
 * @see <a href="http://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix">Wikipedia</a>
 * @since 2.0 (changed to concrete class in 3.0)
 */
 class EigenDecomposition {
  /** Internally used epsilon criteria. */
  companion object {
    private const val EPSILON = 1e-12
  }

  /** Maximum number of iterations accepted in the implicit QL transformation */
  private var maxIter: Int = 30

  /** Main diagonal of the tridiagonal matrix. */
  private lateinit var main: DoubleArray

  /** Secondary diagonal of the tridiagonal matrix. */
  private lateinit var secondary: DoubleArray

  /**
   * Transformer to tridiagonal (may be null if matrix is already
   * tridiagonal).
   */
  private var transformer: TriDiagonalTransformer? = null

  /** Real part of the realEigenvalues. */
  private lateinit var realEigenvalues: DoubleArray

  /** Imaginary part of the realEigenvalues. */
  private lateinit var imagEigenvalues: DoubleArray

  /** Eigenvectors. */
  private lateinit var eigenvectors: Array<ArrayRealVector>

  /** Cached value of V. */
  private var cachedV: MutableMultiArray<Double, D2>? = null

  /** Cached value of D. */
  private var cachedD: MutableMultiArray<Double, D2>? = null

  /** Cached value of Vt. */
  private var cachedVt: MutableMultiArray<Double, D2>? = null

  /** Whether the matrix is symmetric. */
  private val isSymmetric: Boolean

  /**
   * Calculates the eigen decomposition of the given real matrix.
   * <p>
   * Supports decomposition of a general matrix since 3.1.
   *
   * @param matrix Matrix to decompose.
   * @throws MaxCountExceededException if the algorithm fails to converge.
   * @throws MathArithmeticException if the decomposition of a general matrix
   * results in a matrix with zero norm
   * @since 3.1
   */
  @Throws(MathArithmeticException::class)
  constructor(matrix: RealMatrix) {
    val symTol = 10*matrix.rowDimension*matrix.columnDimension*Precision.EPSILON
    isSymmetric = MatrixUtils.isSymmetric(matrix, symTol)
    if (isSymmetric) {
      transformToTridiagonal(matrix)
      findEigenVectors(transformer!!.q!!.getData())
    } else {
      val t = transformToSchur(matrix)
      findEigenVectorsFromSchur(t)
    }
  }

  /**
   * Calculates the eigen decomposition of the symmetric tridiagonal
   * matrix.  The Householder matrix is assumed to be the identity matrix.
   *
   * @param main Main diagonal of the symmetric tridiagonal form.
   * @param secondary Secondary of the tridiagonal form.
   * @throws MaxCountExceededException if the algorithm fails to converge.
   * @since 3.1
   */
  constructor(main: DoubleArray, secondary: DoubleArray) {
    isSymmetric = true
    this.main = main.copyOf()
    this.secondary = secondary.copyOf()
    transformer = null
    val size = main.size
    val z = Array(size) { i -> DoubleArray(size) { j -> if (i == j) 1.0 else 0.0 } }
    findEigenVectors(z)
  }

  /**
   * Gets the matrix V of the decomposition.
   * V is an orthogonal matrix, i.e. its transpose is also its inverse.
   * The columns of V are the eigenvectors of the original matrix.
   * No assumption is made about the orientation of the system axes formed
   * by the columns of V (e.g. in a 3-dimension space, V can form a left-
   * or right-handed system).
   *
   * @return the V matrix.
   */
  val v: RealMatrix get() {
    if (cachedV == null) {
      val m = eigenvectors.size
      cachedV = mk.zeros<Double>(m, m)
      for (k in 0 until m) {
        cachedV!![k] = eigenvectors[k]
      }
    }
    // return the cached matrix
    return cachedV!!
  }

  /**
   * Gets the block diagonal matrix D of the decomposition.
   * D is a block diagonal matrix.
   * Real eigenvalues are on the diagonal while complex values are on
   * 2x2 blocks { {real +imaginary}, {-imaginary, real} }.
   *
   * @return the D matrix.
   *
   * @see #getRealEigenvalues()
   * @see #getImagEigenvalues()
   */
  val d: RealMatrix get() {
    if (cachedD == null) {
      // cache the matrix for subsequent calls
      cachedD = mk.diagonal(realEigenvalues.toList())
      for (i in imagEigenvalues.indices) {
        when {
          Precision.compareTo(imagEigenvalues[i], 0.0, EPSILON) > 0 -> {
            cachedD!![i, i + 1] = imagEigenvalues[i]
          }

          Precision.compareTo(imagEigenvalues[i], 0.0, EPSILON) < 0 -> {
            cachedD!![i, i - 1] = imagEigenvalues[i]
          }
        }
      }
    }
    return cachedD!!
  }

  /**
   * Gets the transpose of the matrix V of the decomposition.
   * V is an orthogonal matrix, i.e. its transpose is also its inverse.
   * The columns of V are the eigenvectors of the original matrix.
   * No assumption is made about the orientation of the system axes formed
   * by the columns of V (e.g. in a 3-dimension space, V can form a left-
   * or right-handed system).
   *
   * @return the transpose of the V matrix.
   */
  val vT: RealMatrix get() {
    if (cachedVt == null) {
      val m = eigenvectors.size
      cachedVt = mk.zeros(m, m)
      for (k in 0 until m) {
        cachedVt!![k] = eigenvectors[k]
      }
    }
    // return the cached matrix
    return cachedVt!!
  }

  /**
   * Returns whether the calculated eigen values are complex or real.
   * <p>The method performs a zero check for each element of the
   * {@link #getImagEigenvalues()} array and returns {@code true} if any
   * element is not equal to zero.
   *
   * @return {@code true} if the eigen values are complex, {@code false} otherwise
   * @since 3.1
   */
  fun hasComplexEigenvalues(): Boolean {
    return imagEigenvalues.any { !Precision.equals(it, 0.0, EPSILON) }
  }

  /**
   * Gets a copy of the real parts of the eigenvalues of the original matrix.
   *
   * @return a copy of the real parts of the eigenvalues of the original matrix.
   *
   * @see #getD()
   * @see #getRealEigenvalue(int)
   * @see #getImagEigenvalues()
   */
  fun getRealEigenvalues(): DoubleArray = realEigenvalues.copyOf()

  /**
   * Returns the real part of the i<sup>th</sup> eigenvalue of the original
   * matrix.
   *
   * @param i index of the eigenvalue (counting from 0)
   * @return real part of the i<sup>th</sup> eigenvalue of the original
   * matrix.
   *
   * @see #getD()
   * @see #getRealEigenvalues()
   * @see #getImagEigenvalue(int)
   */
  fun getRealEigenvalue(i: Int): Double = realEigenvalues[i]

  /**
   * Gets a copy of the imaginary parts of the eigenvalues of the original
   * matrix.
   *
   * @return a copy of the imaginary parts of the eigenvalues of the original
   * matrix.
   *
   * @see #getD()
   * @see #getImagEigenvalue(int)
   * @see #getRealEigenvalues()
   */
  fun getImagEigenvalues(): DoubleArray = imagEigenvalues.copyOf()

  /**
   * Gets the imaginary part of the i<sup>th</sup> eigenvalue of the original
   * matrix.
   *
   * @param i Index of the eigenvalue (counting from 0).
   * @return the imaginary part of the i<sup>th</sup> eigenvalue of the original
   * matrix.
   *
   * @see #getD()
   * @see #getImagEigenvalues()
   * @see #getRealEigenvalue(int)
   */
  fun getImagEigenvalue(i: Int): Double = imagEigenvalues[i]

  /**
   * Gets a copy of the i<sup>th</sup> eigenvector of the original matrix.
   *
   * @param i Index of the eigenvector (counting from 0).
   * @return a copy of the i<sup>th</sup> eigenvector of the original matrix.
   * @see #getD()
   */
  fun getEigenvector(i: Int): RealVector = eigenvectors[i].copy()

  /**
   * Computes the determinant of the matrix.
   *
   * @return the determinant of the matrix.
   */
  fun getDeterminant(): Double {
    return realEigenvalues.fold(1.0) { acc, lambda -> acc*lambda }
  }

  /**
   * Computes the square-root of the matrix.
   * This implementation assumes that the matrix is symmetric and positive
   * definite.
   *
   * @return the square-root of the matrix.
   * @throws MathUnsupportedOperationException if the matrix is not
   * symmetric or not positive definite.
   * @since 3.1
   */
  @Throws(MathUnsupportedOperationException::class)
  fun getSquareRoot(): RealMatrix {
    if (!isSymmetric) {
      throw MathUnsupportedOperationException();
    }
    val sqrtEigenValues = realEigenvalues.map {
      if (it <= 0) throw MathUnsupportedOperationException()
      sqrt(it)
    }.toDoubleArray()
    val sqrtEigen = mk.diagonal(sqrtEigenValues.toList())
    return v dot sqrtEigen dot vT
  }

  /**
   * Gets a solver for finding the A &times; X = B solution in exact
   * linear sense.
   * <p>
   * Since 3.1, eigen decomposition of a general matrix is supported,
   * but the {@link DecompositionSolver} only supports real eigenvalues.
   *
   * @return a solver
   * @throws MathUnsupportedOperationException if the decomposition resulted in
   * complex eigenvalues
   */
  fun getSolver(): Solver {
    if (hasComplexEigenvalues()) throw MathUnsupportedOperationException()
    return Solver(realEigenvalues, imagEigenvalues, eigenvectors)
  }

  /** Specialized solver.
   * @param realEigenvalues Real parts of the eigenvalues.
   * @param imagEigenvalues Imaginary parts of the eigenvalues.
   * @param eigenvectors Eigenvectors.
   */
  class Solver(
    /** Real part of the realEigenvalues. */
    private val realEigenvalues: DoubleArray,
    /** Imaginary part of the realEigenvalues. */
    private val imagEigenvalues: DoubleArray,
    /** Eigenvectors. */
    private val eigenvectors: Array<ArrayRealVector>
  ): ltd.mbor.sciko.linalg.Solver() {

    /**
     * Solves the linear equation A &times; X = B for symmetric matrices A.
     * <p>
     * This method only finds exact linear solutions, i.e. solutions for
     * which ||A &times; X - B|| is exactly 0.
     * </p>
     *
     * @param b Right-hand side of the equation A &times; X = B.
     * @return a Vector X that minimizes the two norm of A &times; X - B.
     *
     * @throws DimensionMismatchException if the matrices dimensions do not match.
     * @throws SingularMatrixException if the decomposed matrix is singular.
     */
    override fun solveVector(b: RealVector): RealVector {
      if (!isNonSingular) throw SingularMatrixException()
      val m = realEigenvalues.size
      if (b.dimension != m) throw DimensionMismatchException(b.dimension, m)
      val bp = DoubleArray(m)
      for (i in 0 until m) {
        val v = eigenvectors[i]
        val s = (v dot b)/realEigenvalues[i]
        for (j in 0 until m) {
          bp[j] += s*v[j]
        }
      }
      return mk.ndarray(bp)
    }

    override fun solveMatrix(b: RealMatrix): RealMatrix {
      if (!isNonSingular) throw SingularMatrixException()
      val m = realEigenvalues.size
      if (b.rowDimension != m) throw DimensionMismatchException(b.rowDimension, m)
      val nColB = b.columnDimension
      val bp = Array(m) { DoubleArray(nColB) }
      val tmpCol = DoubleArray(m)
      for (k in 0 until nColB) {
        for (i in 0 until m) {
          tmpCol[i] = b[i, k]
          bp[i][k] = 0.0
        }
        for (i in 0 until m) {
          val v = eigenvectors[i]
          val s = (v dot mk.ndarray(tmpCol))/realEigenvalues[i]
          for (j in 0 until m) {
            bp[j][k] += s*v[j]
          }
        }
      }
      return mk.ndarray(bp)
    }

    /**
     * Checks whether the decomposed matrix is non-singular.
     *
     * @return true if the decomposed matrix is non-singular.
     */
    val isNonSingular: Boolean get() {
      var largestEigenvalueNorm = 0.0
      // Looping over all values (in case they are not sorted in decreasing
      // order of their norm).
      for (i in realEigenvalues.indices) {
        largestEigenvalueNorm = maxOf(largestEigenvalueNorm, eigenvalueNorm(i))
      }
      // Corner case: zero matrix, all exactly 0 eigenvalues
      if (largestEigenvalueNorm == 0.0) {
        return false
      }
      for (i in realEigenvalues.indices) {
        // Looking for eigenvalues that are 0, where we consider anything much much smaller
        // than the largest eigenvalue to be effectively 0.
        if (Precision.equals(eigenvalueNorm(i) / largestEigenvalueNorm, 0.0, EPSILON)) {
          return false
        }
      }
      return true
    }

    /**
     * @param i which eigenvalue to find the norm of
     * @return the norm of ith (complex) eigenvalue.
     */
    private fun eigenvalueNorm(i: Int): Double {
      val re = realEigenvalues[i]
      val im = imagEigenvalues[i]
      return sqrt(re*re + im*im)
    }

    /**
     * Get the inverse of the decomposed matrix.
     *
     * @return the inverse matrix.
     * @throws SingularMatrixException if the decomposed matrix is singular.
     */
    fun getInverse(): RealMatrix {
      if (!isNonSingular) throw SingularMatrixException()
      val m = realEigenvalues.size
      val invData = Array(m) { DoubleArray(m) }
      for (i in 0 until m) {
        for (j in 0 until m) {
          var invIJ = 0.0
          for (k in 0 until m) {
            val vK = eigenvectors[k].toArray()
            invIJ += vK[i]*vK[j]/realEigenvalues[k]
          }
          invData[i][j] = invIJ
        }
      }
      return mk.ndarray(invData)
    }
  }

  /**
   * Transforms the matrix to tridiagonal form.
   *
   * @param matrix Matrix to transform.
   */
  private fun transformToTridiagonal(matrix: RealMatrix) {
    transformer = TriDiagonalTransformer(matrix)
    main = transformer!!.mainDiagonalRef
    secondary = transformer!!.secondaryDiagonalRef
  }

  /**
   * Find eigenvalues and eigenvectors (Dubrulle et al., 1971)
   *
   * @param householderMatrix Householder matrix of the transformation
   * to tridiagonal form.
   */
  private fun findEigenVectors(householderMatrix: Array<DoubleArray>) {
    val z = householderMatrix.map { it.copyOf() }.toTypedArray()
    val n = main.size
    realEigenvalues = DoubleArray(n)
    imagEigenvalues = DoubleArray(n)
    val e = DoubleArray(n)

    for (i in 0 until n - 1) {
      realEigenvalues[i] = main[i]
      e[i] = secondary[i]
    }
    realEigenvalues[n - 1] = main[n - 1]
    e[n - 1] = 0.0
    // Determine the largest main and secondary value in absolute term.
    var maxAbsoluteValue = 0.0
    for (i in 0 until n) {
      maxAbsoluteValue = maxOf(maxAbsoluteValue, abs(realEigenvalues[i]), abs(e[i]))
    }
    // Make null any main and secondary value too small to be significant
    if (maxAbsoluteValue != 0.0) {
      for (i in 0 until n) {
        if (abs(realEigenvalues[i]) <= Precision.EPSILON * maxAbsoluteValue) {
          realEigenvalues[i] = 0.0
        }
        if (abs(e[i]) <= Precision.EPSILON * maxAbsoluteValue) {
          e[i] = 0.0
        }
      }
    }
    for (j in 0 until n) {
      var its = 0
      var m: Int
      do {
        m = j
        while (m < n - 1) {
          val delta = abs(realEigenvalues[m]) + abs(realEigenvalues[m + 1])
          if (abs(e[m]) + delta == delta) break
          m++
        }
        if (m != j) {
          if (its == maxIter) {
            throw MaxCountExceededException(maxIter)
          }
          its++
          var q = (realEigenvalues[j + 1] - realEigenvalues[j]) / (2 * e[j])
          val t = sqrt(1 + q * q)
          q = if (q < 0) {
            realEigenvalues[m] - realEigenvalues[j] + e[j] / (q - t)
          } else {
            realEigenvalues[m] - realEigenvalues[j] + e[j] / (q + t)
          }
          var u = 0.0
          var s = 1.0
          var c = 1.0
          var i = m - 1
          while(i >= j) {
            val p = s * e[i]
            val h = c * e[i]
            if (abs(p) >= abs(q)) {
              c = q / p
              val t2 = sqrt(c * c + 1.0)
              e[i + 1] = p * t2
              s = 1.0 / t2
              c *= s
            } else {
              s = p / q
              val t2 = sqrt(s * s + 1.0)
              e[i + 1] = q * t2
              c = 1.0 / t2
              s *= c
            }
            if (e[i + 1] == 0.0) {
              realEigenvalues[i + 1] -= u
              e[m] = 0.0
              break
            }
            q = realEigenvalues[i + 1] - u
            val t2 = (realEigenvalues[i] - q) * s + 2.0 * c * h
            u = s * t2
            realEigenvalues[i + 1] = q + u
            q = c * t2 - h
            for (ia in 0 until n) {
              val p2 = z[ia][i + 1]
              z[ia][i + 1] = s * z[ia][i] + c * p2
              z[ia][i] = c * z[ia][i] - s * p2
            }
            i--
          }
          if (t == 0.0 && i >= j) continue
          realEigenvalues[j] -= u
          e[j] = q
          e[m] = 0.0
        }
      } while (m != j)
    }
    //Sort the eigen values (and vectors) in increase order
    for (i in 0 until n) {
      var k = i
      var p = realEigenvalues[i]
      for (j in i + 1 until n) {
        if (realEigenvalues[j] > p) {
          k = j
          p = realEigenvalues[j]
        }
      }
      if (k != i) {
        realEigenvalues[k] = realEigenvalues[i]
        realEigenvalues[i] = p
        for (j in 0 until n) {
          val temp = z[j][i]
          z[j][i] = z[j][k]
          z[j][k] = temp
        }
      }
    }
    // Determine the largest eigen value in absolute term.
    maxAbsoluteValue = realEigenvalues.maxOf { abs(it) }
    if (maxAbsoluteValue != 0.0) {
      for (i in 0 until n) {
        if (abs(realEigenvalues[i]) < Precision.EPSILON * maxAbsoluteValue) {
          realEigenvalues[i] = 0.0
        }
      }
    }

    eigenvectors = Array(n) { i ->
      mk.ndarray(z.map { it[i] }.toDoubleArray())
    }
  }

  /**
   * Transforms the matrix to Schur form and calculates the eigenvalues.
   *
   * @param matrix Matrix to transform.
   * @return the {@link SchurTransformer Shur transform} for this matrix
   */
  private fun transformToSchur(matrix: RealMatrix): SchurTransformer {
    val schurTransform = SchurTransformer(matrix)
    val matT = schurTransform.t.getData()
    realEigenvalues = DoubleArray(matT.size)
    imagEigenvalues = DoubleArray(matT.size)
    for (i in realEigenvalues.indices) {
      if (i == realEigenvalues.lastIndex || Precision.equals(matT[i + 1][i], 0.0, EPSILON)) {
        realEigenvalues[i] = matT[i][i]
      } else {
        val x = matT[i + 1][i + 1]
        val p = 0.5*(matT[i][i] - x)
        val z = sqrt(abs(p*p + matT[i + 1][i]*matT[i][i + 1]))
        realEigenvalues[i] = x + p
        imagEigenvalues[i] = z
        realEigenvalues[i + 1] = x + p
        imagEigenvalues[i + 1] = -z
      }
    }
    return schurTransform
  }

//  /**
//   * Performs a division of two complex numbers.
//   *
//   * @param xr real part of the first number
//   * @param xi imaginary part of the first number
//   * @param yr real part of the second number
//   * @param yi imaginary part of the second number
//   * @return result of the complex division
//   */
//  private fun cdiv(xr: Double, xi: Double, yr: Double, yi: Double): Complex {
//    return Complex(xr, xi).divide(Complex(yr, yi))
//  }
  /**
   * Find eigenvectors from a matrix transformed to Schur form.
   *
   * @param schur the schur transformation of the matrix
   * @throws MathArithmeticException if the Schur form has a norm of zero
   */
  @Throws(MathArithmeticException::class)
  private fun findEigenVectorsFromSchur(schur: SchurTransformer) {
    val matrixT = schur.t.getData()
    val matrixP = schur.p.getData()
    val n = matrixT.size
    // compute matrix norm
    var norm = 0.0
    for (i in 0 until n) {
      for (j in maxOf(i - 1, 0) until n) {
        norm += abs(matrixT[i][j])
      }
    }
    // we can not handle a matrix with zero norm
    if (Precision.equals(norm, 0.0, EPSILON)) {
      throw MathArithmeticException()
    }
    // Backsubstitute to find vectors of upper triangular form
    var r = 0.0
    var s = 0.0
    var z = 0.0
    for (idx in n - 1 downTo 0) {
      val p = realEigenvalues[idx]
      var q = imagEigenvalues[idx]
      if (Precision.equals(q, 0.0)) {
        // Real vector
        var l = idx
        matrixT[idx][idx] = 1.0
        for (i in idx - 1 downTo 0) {
          val w = matrixT[i][i] - p
          r = 0.0
          for (j in l..idx) {
            r += matrixT[i][j] * matrixT[j][idx]
          }
          if (Precision.compareTo(imagEigenvalues[i], 0.0, EPSILON) < 0) {
            z = w
            s = r
          } else {
            l = i
            if (Precision.equals(imagEigenvalues[i], 0.0)) {
              matrixT[i][idx] = if (w != 0.0) -r / w else -r / (Precision.EPSILON * norm)
            } else {
              // Solve real equations
              val x = matrixT[i][i + 1]
              val y = matrixT[i + 1][i]
              q = (realEigenvalues[i] - p) * (realEigenvalues[i] - p) + imagEigenvalues[i] * imagEigenvalues[i]
              val t = (x * s - z * r) / q
              matrixT[i][idx] = t
              matrixT[i + 1][idx] = if (abs(x) > abs(z)) {
                (-r - w * t) / x
              } else {
                (-s - y * t) / z
              }
            }
            // Overflow control
            val t = abs(matrixT[i][idx])
            if ((Precision.EPSILON * t) * t > 1) {
              for (j in i..idx) {
                matrixT[j][idx] /= t
              }
            }
          }
        }
      } else if (q < 0.0) {
      // Complex vector
        val l = idx - 1
      // Last vector component imaginary so matrix is triangular
        if (abs(matrixT[idx][idx - 1]) > abs(matrixT[idx - 1][idx])) {
          matrixT[idx - 1][idx - 1] = q / matrixT[idx][idx - 1]
          matrixT[idx - 1][idx] = -(matrixT[idx][idx] - p) / matrixT[idx][idx - 1]
        } else {
          val result = ComplexDouble(0.0, -matrixT[idx - 1][idx]) / ComplexDouble(matrixT[idx - 1][idx - 1] - p, q)
          matrixT[idx - 1][idx - 1] = result.re
          matrixT[idx - 1][idx] = result.im
        }
        matrixT[idx][idx - 1] = 0.0
        matrixT[idx][idx] = 1.0
        for (i in idx - 2 downTo 0) {
          var ra = 0.0
          var sa = 0.0
          for (j in l..idx) {
            ra += matrixT[i][j] * matrixT[j][idx - 1]
            sa += matrixT[i][j] * matrixT[j][idx]
          }
          val w = matrixT[i][i] - p
          if (Precision.compareTo(imagEigenvalues[i], 0.0, EPSILON) < 0) {
            z = w
            r = ra
            s = sa
          } else {
            if (Precision.equals(imagEigenvalues[i], 0.0)) {
              val c = ComplexDouble(-ra, -sa).div(ComplexDouble(w, q))
              matrixT[i][idx - 1] = c.re
              matrixT[i][idx] = c.im
            } else {
              // Solve complex equations
              val x = matrixT[i][i + 1]
              val y = matrixT[i + 1][i]
              val vr = (realEigenvalues[i] - p) * (realEigenvalues[i] - p) +
                imagEigenvalues[i] * imagEigenvalues[i] - q * q
              val vi = (realEigenvalues[i] - p) * 2.0 * q
              val adjustedVr = if (Precision.equals(vr, 0.0) && Precision.equals(vi, 0.0)) {
                Precision.EPSILON * norm * (abs(w) + abs(q) + abs(x) +
                  abs(y) + abs(z))
              } else vr
              val c = ComplexDouble(x * r - z * ra + q * sa, x * s - z * sa - q * ra)
                .div(ComplexDouble(adjustedVr, vi))
              matrixT[i][idx - 1] = c.re
              matrixT[i][idx] = c.im
              if (abs(x) > (abs(z) + abs(q))) {
                matrixT[i + 1][idx - 1] = (-ra - w * matrixT[i][idx - 1] + q * matrixT[i][idx]) / x
                matrixT[i + 1][idx] = (-sa - w * matrixT[i][idx] - q * matrixT[i][idx - 1]) / x
              } else {
                val c2 = ComplexDouble(-r - y * matrixT[i][idx - 1], -s - y * matrixT[i][idx])
                  .div(ComplexDouble(z, q))
                matrixT[i + 1][idx - 1] = c2.re
                matrixT[i + 1][idx] = c2.im
              }
            }
            // Overflow control
            val t = maxOf(abs(matrixT[i][idx - 1]), abs(matrixT[i][idx]))
            if ((Precision.EPSILON * t) * t > 1) {
              for (j in i..idx) {
                matrixT[j][idx - 1] /= t
                matrixT[j][idx] /= t
              }
            }
          }
        }
      }
    }
    // Back transformation to get eigenvectors of original matrix
    for (j in n - 1 downTo 0) {
      for (i in 0 until n) {
        z = 0.0
        for (k in 0..minOf(j, n - 1)) {
          z += matrixP[i][k] * matrixT[k][j]
        }
        matrixP[i][j] = z
      }
    }

    eigenvectors = Array(n) { i ->
      mk.ndarray(DoubleArray(n) { j -> matrixP[j][i] })
    }
  }
}
