package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray


/**
 * Calculates the LUP-decomposition of a square matrix.
 *
 * The LUP-decomposition of a matrix A consists of three matrices L, U and
 * P that satisfy: PA = LU. L is lower triangular (with unit
 * diagonal terms), U is upper triangular and P is a permutation matrix. All
 * matrices are mm.
 *
 * As shown by the presence of the P matrix, this decomposition is
 * implemented using partial pivoting.
 *
 * This class is based on the class with similar name from the
 * [JAMA](http://math.nist.gov/javanumerics/jama/) library.
 *
 *  * a [getP][.getP] method has been added,
 *  * the `det` method has been renamed as [   getDeterminant][.getDeterminant],
 *  * the `getDoublePivot` method has been removed (but the int based
 * [getPivot][.getPivot] method has been kept),
 *  * the `solve` and `isNonSingular` methods have been replaced
 * by a [getSolver][.getSolver] method and the equivalent methods
 * provided by the returned [DecompositionSolver].
 *
 *
 * @see [MathWorld](http://mathworld.wolfram.com/LUDecomposition.html)
 *
 * @see [Wikipedia](http://en.wikipedia.org/wiki/LU_decomposition)
 *
 * @since 2.0 (changed to concrete class in 3.0)
 */
class LUDecomposition @JvmOverloads constructor(matrix: RealMatrix, singularityThreshold: Double = DEFAULT_TOO_SMALL) {
  /** Entries of LU decomposition.  */
  private val lu: Array<DoubleArray>

  /** Pivot permutation associated with LU decomposition.  */
  internal val pivot: IntArray

  /** Parity of the permutation associated with the LU decomposition.  */
  private var even: Boolean

  /** Singularity indicator.  */
  private var singular: Boolean

  /** Cached value of L.  */
  private var cachedL: RealMatrix?

  /** Cached value of U.  */
  private var cachedU: RealMatrix?

  /** Cached value of P.  */
  private var cachedP: RealMatrix?
  /**
   * Calculates the LU-decomposition of the given matrix.
   * @param matrix The matrix to decompose.
   * @param singularityThreshold threshold (based on partial row norm)
   * under which a matrix is considered singular
   * @throws NonSquareMatrixException if matrix is not square
   */
  /**
   * Calculates the LU-decomposition of the given matrix.
   * This constructor uses 1e-11 as default value for the singularity
   * threshold.
   *
   * @param matrix Matrix to decompose.
   * @throws NonSquareMatrixException if matrix is not square.
   */
  init {
    if (!matrix.isSquare) {
      throw NonSquareMatrixException(
        matrix.rowDimension,
        matrix.columnDimension
      )
    }
    val m = matrix.columnDimension
    lu = matrix.toArray()
    pivot = IntArray(m)
    cachedL = null
    cachedU = null
    cachedP = null
    // Initialize permutation array and parity
    for (row in 0..<m) {
      pivot[row] = row
    }
    even = true
    singular = false
    // Loop over columns
    for (col in 0..<m) {
      // upper
      for (row in 0..<col) {
        val luRow = lu[row]
        var sum = luRow[col]
        for (i in 0..<row) {
          sum -= luRow[i] * lu[i][col]
        }
        luRow[col] = sum
      }
      // lower
      var max = col // permutation row
      var largest = Double.NEGATIVE_INFINITY
      for (row in col..<m) {
        val luRow = lu[row]
        var sum = luRow[col]
        for (i in 0..<col) {
          sum -= luRow[i] * lu[i][col]
        }
        luRow[col] = sum
        // maintain best permutation choice
        if (FastMath.abs(sum) > largest) {
          largest = FastMath.abs(sum)
          max = row
        }
      }
      // Singularity check
      if (FastMath.abs(lu[max][col]) < singularityThreshold) {
        singular = true
      } else {
        // Pivot if necessary
        if (max != col) {
          var tmp = 0.0
          val luMax = lu[max]
          val luCol = lu[col]
          for (i in 0..<m) {
            tmp = luMax[i]
            luMax[i] = luCol[i]
            luCol[i] = tmp
          }
          val temp = pivot[max]
          pivot[max] = pivot[col]
          pivot[col] = temp
          even = !even
        }
        // Divide the lower elements by the "winning" diagonal elt.
        val luDiag = lu[col][col]
        for (row in col + 1..<m) {
          lu[row][col] /= luDiag
        }
      }
    }
  }

  val l: RealMatrix?
    /**
     * Returns the matrix L of the decomposition.
     *
     * L is a lower-triangular matrix
     * @return the L matrix (or null if decomposed matrix is singular)
     */
    get() {
      if ((cachedL == null) && !singular) {
        val m = pivot.size
        cachedL = mk.zeros<Double>(m, m).also {
          for (i in 0..<m) {
            val luI = lu[i]
            for (j in 0..<i) {
              it.set(i, j, luI[j])
            }
            it.set(i, i, 1.0)
          }
        }
      }
      return cachedL
    }
  val u: RealMatrix?
    /**
     * Returns the matrix U of the decomposition.
     *
     * U is an upper-triangular matrix
     * @return the U matrix (or null if decomposed matrix is singular)
     */
    get() {
      if ((cachedU == null) && !singular) {
        val m = pivot.size
        cachedU = mk.zeros<Double>(m, m).also {
          for (i in 0..<m) {
            val luI = lu[i]
            for (j in i..<m) {
              it.set(i, j, luI[j])
            }
          }
        }
      }
      return cachedU
    }
  val p: RealMatrix?
    /**
     * Returns the P rows permutation matrix.
     *
     * P is a sparse matrix with exactly one element set to 1.0 in
     * each row and each column, all other elements being set to 0.0.
     *
     * The positions of the 1 elements are given by the [ pivot permutation vector][.getPivot].
     * @return the P rows permutation matrix (or null if decomposed matrix is singular)
     * @see .getPivot
     */
    get() {
      if ((cachedP == null) && !singular) {
        val m = pivot.size
        cachedP = mk.zeros<Double>(m, m).also {
          for (i in 0..<m) {
            it.set(i, pivot[i], 1.0)
          }
        }
      }
      return cachedP
    }

  /**
   * Returns the pivot permutation vector.
   * @return the pivot permutation vector
   * @see .getP
   */
  fun getPivot(): IntArray {
    return pivot.clone()
  }

  val determinant: Double
    /**
     * Return the determinant of the matrix
     * @return determinant of the matrix
     */
    get() {
      if (singular) {
        return 0.0
      } else {
        val m = pivot.size
        var determinant = (if (even) 1 else -1).toDouble()
        for (i in 0..<m) {
          determinant *= lu[i][i]
        }
        return determinant
      }
    }
  val solver: Solver
    /**
     * Get a solver for finding the A  X = B solution in exact linear
     * sense.
     * @return a solver
     */
    get() = Solver(lu, pivot, singular)

  /** Specialized solver.  */
  class Solver
  /**
   * Build a solver from decomposed matrix.
   * @param lu entries of LU decomposition
   * @param pivot pivot permutation associated with LU decomposition
   * @param singular singularity indicator
   */(
    /** Entries of LU decomposition.  */
    private val lu: Array<DoubleArray>,
    /** Pivot permutation associated with LU decomposition.  */
    private val pivot: IntArray,
    /** Singularity indicator.  */
    private val singular: Boolean
  ) {
    /** {@inheritDoc}  */
    val isNonSingular: Boolean get() = !singular

    /** {@inheritDoc}  */
    @JvmName("solveRealVector")
    fun solve(b: RealVector): RealVector {
      val m = pivot.size
      if (b.dimension != m) {
        throw DimensionMismatchException(b.dimension, m)
      }
      if (singular) {
        throw SingularMatrixException()
      }
      val bp = DoubleArray(m)
      // Apply permutations to b
      for (row in 0..<m) {
        bp[row] = b[pivot[row]]
      }
      // Solve LY = b
      for (col in 0..<m) {
        val bpCol = bp[col]
        for (i in col + 1..<m) {
          bp[i] -= bpCol * lu[i][col]
        }
      }
      // Solve UX = Y
      for (col in m - 1 downTo 0) {
        bp[col] /= lu[col][col]
        val bpCol = bp[col]
        for (i in 0..<col) {
          bp[i] -= bpCol * lu[i][col]
        }
      }
      return mk.ndarray(bp)
    }

    /** {@inheritDoc}  */
    @JvmName("solveRealMatrix")
    fun solve(b: RealMatrix): RealMatrix {
      val m = pivot.size
      if (b.rowDimension != m) {
        throw DimensionMismatchException(b.rowDimension, m)
      }
      if (singular) {
        throw SingularMatrixException()
      }
      val nColB = b.columnDimension
      // Apply permutations to b
      val bp = Array(m) { DoubleArray(nColB) }
      for (row in 0..<m) {
        val bpRow = bp[row]
        val pRow = pivot[row]
        for (col in 0..<nColB) {
          bpRow[col] = b[pRow, col]
        }
      }
      // Solve LY = b
      for (col in 0..<m) {
        val bpCol = bp[col]
        for (i in col + 1..<m) {
          val bpI = bp[i]
          val luICol = lu[i][col]
          for (j in 0..<nColB) {
            bpI[j] -= bpCol[j] * luICol
          }
        }
      }
      // Solve UX = Y
      for (col in m - 1 downTo 0) {
        val bpCol = bp[col]
        val luDiag = lu[col][col]
        for (j in 0..<nColB) {
          bpCol[j] /= luDiag
        }
        for (i in 0..<col) {
          val bpI = bp[i]
          val luICol = lu[i][col]
          for (j in 0..<nColB) {
            bpI[j] -= bpCol[j] * luICol
          }
        }
      }
      return mk.ndarray(bp)
    }

    /**
     * Get the inverse of the decomposed matrix.
     *
     * @return the inverse matrix.
     * @throws SingularMatrixException if the decomposed matrix is singular.
     */
    fun getInverse(): RealMatrix {
      return solve(mk.identity(pivot.size))
    }
  }

  companion object {
    /** Default bound to determine effective singularity in LU decomposition.  */
    private const val DEFAULT_TOO_SMALL = 1e-11
  }
}
