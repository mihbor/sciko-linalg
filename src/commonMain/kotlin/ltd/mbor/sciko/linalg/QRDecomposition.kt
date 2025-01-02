package ltd.mbor.sciko.linalg

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import java.util.*

/**
 * Calculates the QR-decomposition of a matrix.
 *
 * The QR-decomposition of a matrix A consists of two matrices Q and R
 * that satisfy: A = QR, Q is orthogonal (Q<sup>T</sup>Q = I), and R is
 * upper triangular. If A is mn, Q is mm and R mn.
 *
 * This class compute the decomposition using Householder reflectors.
 *
 * For efficiency purposes, the decomposition in packed form is transposed.
 * This allows inner loop to iterate inside rows, which is much more cache-efficient
 * in Java.
 *
 * This class is based on the class with similar name from the
 * [JAMA](http://math.nist.gov/javanumerics/jama/) library, with the
 * following changes:
 *
 *  * a [getQT][.getQT] method has been added,
 *  * the `solve` and `isFullRank` methods have been replaced
 * by a [getSolver][.getSolver] method and the equivalent methods
 * provided by the returned [DecompositionSolver].
 *
 *
 * @see [MathWorld](http://mathworld.wolfram.com/QRDecomposition.html)
 *
 * @see [Wikipedia](http://en.wikipedia.org/wiki/QR_decomposition)
 *
 *
 * @since 1.2 (changed to concrete class in 3.0)
 */
open class QRDecomposition @JvmOverloads constructor(
  matrix: RealMatrix,
  /** Singularity threshold.  */
  private val threshold: Double = 0.0
) {
  /**
   * A packed TRANSPOSED representation of the QR decomposition.
   *
   * The elements BELOW the diagonal are the elements of the UPPER triangular
   * matrix R, and the rows ABOVE the diagonal are the Householder reflector vectors
   * from which an explicit form of Q can be recomputed if desired.
   */
  private val qrt: Array<DoubleArray>

  /** The diagonal elements of R.  */
  private val rDiag: DoubleArray

  /** Cached value of Q.  */
  private var cachedQ: RealMatrix?

  /** Cached value of QT.  */
  private var cachedQT: RealMatrix?

  /** Cached value of R.  */
  private var cachedR: RealMatrix?

  /** Cached value of H.  */
  private var cachedH: RealMatrix?
  /**
   * Calculates the QR-decomposition of the given matrix.
   *
   * @param matrix The matrix to decompose.
   * @param threshold Singularity threshold.
   */
  /**
   * Calculates the QR-decomposition of the given matrix.
   * The singularity threshold defaults to zero.
   *
   * @param matrix The matrix to decompose.
   *
   * @see .QRDecomposition
   */
  init {
    val m: Int = matrix.rowDimension
    val n: Int = matrix.columnDimension
    qrt = matrix.transpose().getData()
    rDiag = DoubleArray(FastMath.min(m, n))
    cachedQ = null
    cachedQT = null
    cachedR = null
    cachedH = null
    decompose(qrt)
  }

  /** Decompose matrix.
   * @param matrix transposed matrix
   * @since 3.2
   */
  protected open fun decompose(matrix: Array<DoubleArray>) {
    for (minor in 0 until FastMath.min(matrix.size, matrix[0].size)) {
      performHouseholderReflection(minor, matrix)
    }
  }

  /** Perform Householder reflection for a minor A(minor, minor) of A.
   * @param minor minor index
   * @param matrix transposed matrix
   * @since 3.2
   */
  protected open fun performHouseholderReflection(minor: Int, matrix: Array<DoubleArray>) {
    val qrtMinor = matrix[minor]
    /*
         * Let x be the first column of the minor, and a^2 = |x|^2.
         * x will be in the positions qr[minor][minor] through qr[m][minor].
         * The first column of the transformed minor will be (a,0,0,..)'
         * The sign of a is chosen to be opposite to the sign of the first
         * component of x. Let's find a:
         */
    var xNormSqr = 0.0
    for (row in minor until qrtMinor.size) {
      val c = qrtMinor[row]
      xNormSqr += c * c
    }
    val a: Double = if ((qrtMinor[minor] > 0)) -FastMath.sqrt(xNormSqr) else FastMath.sqrt(xNormSqr)
    rDiag[minor] = a
    if (a != 0.0) {
      /*
             * Calculate the normalized reflection vector v and transform
             * the first column. We know the norm of v beforehand: v = x-ae
             * so |v|^2 = <x-ae,x-ae> = <x,x>-2a<x,e>+a^2<e,e> =
             * a^2+a^2-2a<x,e> = 2a*(a - <x,e>).
             * Here <x, e> is now qr[minor][minor].
             * v = x-ae is stored in the column at qr:
             */
      qrtMinor[minor] -= a // now |v|^2 = -2a*(qr[minor][minor])
      /*
             * Transform the rest of the columns of the minor:
             * They will be transformed by the matrix H = I-2vv'/|v|^2.
             * If x is a column vector of the minor, then
             * Hx = (I-2vv'/|v|^2)x = x-2vv'x/|v|^2 = x - 2<x,v>/|v|^2 v.
             * Therefore the transformation is easily calculated by
             * subtracting the column vector (2<x,v>/|v|^2)v from x.
             *
             * Let 2<x,v>/|v|^2 = alpha. From above we have
             * |v|^2 = -2a*(qr[minor][minor]), so
             * alpha = -<x,v>/(a*qr[minor][minor])
             */
      for (col in minor + 1 until matrix.size) {
        val qrtCol = matrix[col]
        var alpha = 0.0
        for (row in minor until qrtCol.size) {
          alpha -= qrtCol[row] * qrtMinor[row]
        }
        alpha /= a * qrtMinor[minor]
        // Subtract the column vector alpha*v from x.
        for (row in minor until qrtCol.size) {
          qrtCol[row] -= alpha * qrtMinor[row]
        }
      }
    }
  }

  val r: RealMatrix
    /**
     * Returns the matrix R of the decomposition.
     *
     * R is an upper-triangular matrix
     * @return the R matrix
     */
    get() {
      if (cachedR == null) {
        // R is supposed to be m x n
        val n = qrt.size
        val m = qrt[0].size
        val ra = Array(m) { DoubleArray(n) }
        // copy the diagonal from rDiag and the upper triangle of qr
        for (row in FastMath.min(m, n) - 1 downTo 0) {
          ra[row][row] = rDiag[row]
          for (col in row + 1 until n) {
            ra[row][col] = qrt[col][row]
          }
        }
        cachedR = mk.ndarray(ra)
      }
      // return the cached matrix
      return cachedR!!
    }
  val q: RealMatrix
    /**
     * Returns the matrix Q of the decomposition.
     *
     * Q is an orthogonal matrix
     * @return the Q matrix
     */
    get() {
      if (cachedQ == null) {
        cachedQ = qT.transpose()
      }
      return cachedQ!!
    }
  val qT: RealMatrix
    /**
     * Returns the transpose of the matrix Q of the decomposition.
     *
     * Q is an orthogonal matrix
     * @return the transpose of the Q matrix, Q<sup>T</sup>
     */
    get() {
      if (cachedQT == null) {
        // QT is supposed to be m x m
        val n = qrt.size
        val m = qrt[0].size
        val qta = Array(m) { DoubleArray(m) }
        /*
              * Q = Q1 Q2 ... Q_m, so Q is formed by first constructing Q_m and then
              * applying the Householder transformations Q_(m-1),Q_(m-2),...,Q1 in
              * succession to the result
              */
        for (minor in m - 1 downTo FastMath.min(m, n)) {
          qta[minor][minor] = 1.0
        }
        for (minor in FastMath.min(m, n) - 1 downTo 0) {
          val qrtMinor = qrt[minor]
          qta[minor][minor] = 1.0
          if (qrtMinor[minor] != 0.0) {
            for (col in minor until m) {
              var alpha = 0.0
              for (row in minor until m) {
                alpha -= qta[col][row] * qrtMinor[row]
              }
              alpha /= rDiag[minor] * qrtMinor[minor]
              for (row in minor until m) {
                qta[col][row] += -alpha * qrtMinor[row]
              }
            }
          }
        }
        cachedQT = mk.ndarray(qta)
      }
      // return the cached matrix
      return cachedQT!!
    }
  val h: RealMatrix
    /**
     * Returns the Householder reflector vectors.
     *
     * H is a lower trapezoidal matrix whose columns represent
     * each successive Householder reflector vector. This matrix is used
     * to compute Q.
     * @return a matrix containing the Householder reflector vectors
     */
    get() {
      if (cachedH == null) {
        val n = qrt.size
        val m = qrt[0].size
        val ha = Array(m) { DoubleArray(n) }
        for (i in 0 until m) {
          for (j in 0 until FastMath.min(i + 1, n)) {
            ha[i][j] = qrt[j][i] / -rDiag[j]
          }
        }
        cachedH = mk.ndarray(ha)
      }
      // return the cached matrix
      return cachedH!!
    }
  open val solver: Solver
    /**
     * Get a solver for finding the A  X = B solution in least square sense.
     *
     *
     * Least Square sense means a solver can be computed for an overdetermined system,
     * (i.e. a system with more equations than unknowns, which corresponds to a tall A
     * matrix with more rows than columns). In any case, if the matrix is singular
     * within the tolerance set at [construction][QRDecomposition.QRDecomposition], an error will be triggered when
     * the [solve][DecompositionSolver.solve] method will be called.
     *
     * @return a solver
     */
    get() = Solver(qrt, rDiag, threshold)

  /** Specialized solver.  */
  class Solver
  /**
   * Build a solver from decomposed matrix.
   *
   * @param qrt Packed TRANSPOSED representation of the QR decomposition.
   * @param rDiag Diagonal elements of R.
   * @param threshold Singularity threshold.
   */(
    /**
     * A packed TRANSPOSED representation of the QR decomposition.
     *
     * The elements BELOW the diagonal are the elements of the UPPER triangular
     * matrix R, and the rows ABOVE the diagonal are the Householder reflector vectors
     * from which an explicit form of Q can be recomputed if desired.
     */
    private val qrt: Array<DoubleArray>,
    /** The diagonal elements of R.  */
    private val rDiag: DoubleArray,
    /** Singularity threshold.  */
    private val threshold: Double
  ) {
    val isNonSingular: Boolean
      /** {@inheritDoc}  */
      get() {
        for (diag in rDiag) {
          if (FastMath.abs(diag) <= threshold) {
            return false
          }
        }
        return true
      }

    @JvmName("solveRealVector")
    fun solve(b: RealVector): RealVector {
      val n = qrt.size
      val m = qrt[0].size
      if (b.dimension != m) {
        throw DimensionMismatchException(b.dimension, m)
      }
      if (!isNonSingular) {
        throw SingularMatrixException()
      }
      val x = DoubleArray(n)
      val y: DoubleArray = b.toArray()
      // apply Householder transforms to solve Q.y = b
      for (minor in 0 until FastMath.min(m, n)) {
        val qrtMinor = qrt[minor]
        var dotProduct = 0.0
        for (row in minor until m) {
          dotProduct += y[row] * qrtMinor[row]
        }
        dotProduct /= rDiag[minor] * qrtMinor[minor]
        for (row in minor until m) {
          y[row] += dotProduct * qrtMinor[row]
        }
      }
      // solve triangular system R.x = y
      for (row in rDiag.indices.reversed()) {
        y[row] /= rDiag[row]
        val yRow = y[row]
        val qrtRow = qrt[row]
        x[row] = yRow
        for (i in 0 until row) {
          y[i] -= yRow * qrtRow[i]
        }
      }
      return mk.ndarray(x)
    }

    @JvmName("solveRealMatrix")
    fun solve(b: RealMatrix): RealMatrix {
      val n = qrt.size
      val m = qrt[0].size
      if (b.rowDimension != m) {
        throw DimensionMismatchException(b.rowDimension, m)
      }
      if (!isNonSingular) {
        throw SingularMatrixException()
      }
      val columns: Int = b.columnDimension
      val blockSize: Int = BLOCK_SIZE
      val cBlocks = (columns + blockSize - 1) / blockSize
      val xBlocks = createBlocksLayout(n, columns)
//      val y = mk.zeros<Double>(b.getRowDimension(), blockSize)
      val alpha = DoubleArray(blockSize)
      for (kBlock in 0 until cBlocks) {
        val kStart = kBlock * blockSize
        val kEnd: Int = FastMath.min(kStart + blockSize, columns)
        val kWidth = kEnd - kStart
        // get the right hand side vector
//        b.copySubMatrix(0, m - 1, kStart, kEnd - 1, y)
        val y = mk.ndarray(b[0..m-1, kStart..kEnd-1].toArray())
        // apply Householder transforms to solve Q.y = b
        for (minor in 0 until FastMath.min(m, n)) {
          val qrtMinor = qrt[minor]
          val factor = 1.0 / (rDiag[minor] * qrtMinor[minor])
          Arrays.fill(alpha, 0, kWidth, 0.0)
          for (row in minor until m) {
            val d = qrtMinor[row]
            val yRow = y[row]
            for (k in 0 until kWidth) {
              alpha[k] += d * yRow[k]
            }
          }
          for (k in 0 until kWidth) {
            alpha[k] *= factor
          }
          for (row in minor until m) {
            val d = qrtMinor[row]
            for (k in 0 until kWidth) {
              y[row, k] += alpha[k] * d
            }
          }
        }
        // solve triangular system R.x = y
        for (j in rDiag.indices.reversed()) {
          val jBlock = j / blockSize
          val jStart = jBlock * blockSize
          val factor = 1.0 / rDiag[j]
          var index = (j - jStart) * kWidth
          for (k in 0 until kWidth) {
            y[j, k] *= factor
            xBlocks[jBlock * cBlocks + kBlock]!![index++] = y[j][k]
          }
          val qrtJ = qrt[j]
          for (i in 0 until j) {
            val rIJ = qrtJ[i]
            for (k in 0 until kWidth) {
              y[i, k] -= y[j][k] * rIJ
            }
          }
        }
      }
      return mk.ndarray(xBlocks.getData(n, columns)).also { println("solve: $it") }
    }

    val inverse: RealMatrix
      /**
       * {@inheritDoc}
       * @throws SingularMatrixException if the decomposed matrix is singular.
       */
      get() = solve(mk.identity(qrt[0].size))

    companion object {
      const val BLOCK_SIZE = 52
    }

    fun createBlocksLayout(rows: Int, columns: Int): Array<DoubleArray?> {
      val blockRows: Int = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE
      val blockColumns: Int = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE
      val blocks = arrayOfNulls<DoubleArray>(blockRows * blockColumns)
      var blockIndex = 0
      for (iBlock in 0 until blockRows) {
        val pStart: Int = iBlock * BLOCK_SIZE
        val pEnd = FastMath.min(pStart + BLOCK_SIZE, rows)
        val iHeight = pEnd - pStart
        for (jBlock in 0 until blockColumns) {
          val qStart: Int = jBlock * BLOCK_SIZE
          val qEnd = FastMath.min(qStart + BLOCK_SIZE, columns)
          val jWidth = qEnd - qStart
          blocks[blockIndex] = DoubleArray(iHeight * jWidth)
          ++blockIndex
        }
      }
      return blocks
    }

    fun Array<DoubleArray?>.getData(rows: Int, columns: Int): Array<DoubleArray> {
      println("getData of (${this[0]!!.size}) $rows x $columns}")
      val blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
      val blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
      val data = Array(rows) { DoubleArray(columns) }
      val lastColumns: Int = columns - (blockColumns - 1) * BLOCK_SIZE
      for (iBlock in 0 until blockRows) {
        val pStart = iBlock * BLOCK_SIZE
        val pEnd = FastMath.min(pStart + BLOCK_SIZE, rows)
        var regularPos = 0
        var lastPos = 0
        for (p in pStart until pEnd) {
          val dataP = data[p]
          var blockIndex: Int = iBlock * blockColumns
          var dataPos = 0
          for (jBlock in 0 until blockColumns - 1) {
            System.arraycopy(get(blockIndex++), regularPos, dataP, dataPos, BLOCK_SIZE)
            dataPos += BLOCK_SIZE
          }
          System.arraycopy(get(blockIndex), lastPos, dataP, dataPos, lastColumns)
          regularPos += BLOCK_SIZE
          lastPos += lastColumns
        }
      }
      return data
    }
  }
}

class DimensionMismatchException(rowDimension: Int, columnDimension: Int) : Throwable()

class SingularMatrixException : Throwable()
