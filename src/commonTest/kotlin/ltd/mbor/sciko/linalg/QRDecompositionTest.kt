package ltd.mbor.sciko.linalg

import ltd.mbor.sciko.linalg.QRDecomposition.Solver.Companion.BLOCK_SIZE
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

fun RealMatrix.getNorm() = mk.linalg.norm(this)

class QRDecompositionTest {
  private val testData3x3NonSingular = arrayOf(
    doubleArrayOf(12.0, -51.0, 4.0),
    doubleArrayOf(6.0, 167.0, -68.0),
    doubleArrayOf(-4.0, 24.0, -41.0),
  )

  private val testData3x3Singular = arrayOf(
    doubleArrayOf(1.0, 4.0, 7.0),
    doubleArrayOf(2.0, 5.0, 8.0),
    doubleArrayOf(3.0, 6.0, 9.0),
  )

  private val testData3x4 = arrayOf(
    doubleArrayOf(12.0, -51.0, 4.0, 1.0),
    doubleArrayOf(6.0, 167.0, -68.0, 2.0),
    doubleArrayOf(-4.0, 24.0, -41.0, 3.0),
  )

  private val testData4x3 = arrayOf(
    doubleArrayOf(12.0, -51.0, 4.0),
    doubleArrayOf(6.0, 167.0, -68.0),
    doubleArrayOf(-4.0, 24.0, -41.0),
    doubleArrayOf(-5.0, 34.0, 7.0),
  )

  /** test dimensions  */
  @Test
  fun testDimensions() {
    checkDimension(mk.ndarray(testData3x3NonSingular))

    checkDimension(mk.ndarray(testData4x3))

    checkDimension(mk.ndarray(testData3x4))

    val r = Random(643895747384642L)
    val p: Int = (5 * BLOCK_SIZE) / 4
    val q: Int = (7 * BLOCK_SIZE) / 4
    checkDimension(createTestMatrix(r, p, q))
    checkDimension(createTestMatrix(r, q, p))
  }

  private fun checkDimension(m: RealMatrix) {
    val rows = m.rowDimension
    val columns = m.columnDimension
    val qr = QRDecomposition(m)
    assertEquals(rows, qr.q.rowDimension)
    assertEquals(rows, qr.q.columnDimension)
    assertEquals(rows, qr.r.rowDimension)
    assertEquals(columns, qr.r.columnDimension)
  }

  /** test A = QR  */
  @Test
  fun testAEqualQR() {
    checkAEqualQR(mk.ndarray(testData3x3NonSingular))

    checkAEqualQR(mk.ndarray(testData3x3Singular))

    checkAEqualQR(mk.ndarray(testData3x4))

    checkAEqualQR(mk.ndarray(testData4x3))

    val r = Random(643895747384642L)
    val p: Int = (5 * BLOCK_SIZE) / 4
    val q: Int = (7 * BLOCK_SIZE) / 4
    checkAEqualQR(createTestMatrix(r, p, q))

    checkAEqualQR(createTestMatrix(r, q, p))
  }

  private fun checkAEqualQR(m: RealMatrix) {
    val qr: QRDecomposition = QRDecomposition(m)
    val norm: Double = qr.q.dot(qr.r).minus(m).getNorm()
    assertEquals(0.0, norm, normTolerance)
  }

  /** test the orthogonality of Q  */
  @Test
  fun testQOrthogonal() {
    checkQOrthogonal(mk.ndarray(testData3x3NonSingular))

    checkQOrthogonal(mk.ndarray(testData3x3Singular))

    checkQOrthogonal(mk.ndarray(testData3x4))

    checkQOrthogonal(mk.ndarray(testData4x3))

    val r = Random(643895747384642L)
    val p: Int = (5 * BLOCK_SIZE) / 4
    val q: Int = (7 * BLOCK_SIZE) / 4
    checkQOrthogonal(createTestMatrix(r, p, q))

    checkQOrthogonal(createTestMatrix(r, q, p))
  }

  private fun checkQOrthogonal(m: RealMatrix) {
    val qr: QRDecomposition = QRDecomposition(m)
    val eye: RealMatrix = mk.identity(m.rowDimension)
    val norm: Double = qr.qT.dot(qr.q).minus(eye).getNorm()
    assertEquals(0.0, norm, normTolerance)
  }

  /** test that R is upper triangular  */
  @Test
  fun testRUpperTriangular() {
    var matrix: RealMatrix = mk.ndarray(testData3x3NonSingular)
    checkUpperTriangular(QRDecomposition(matrix).r)

    matrix = mk.ndarray(testData3x3Singular)
    checkUpperTriangular(QRDecomposition(matrix).r)

    matrix = mk.ndarray(testData3x4)
    checkUpperTriangular(QRDecomposition(matrix).r)

    matrix = mk.ndarray(testData4x3)
    checkUpperTriangular(QRDecomposition(matrix).r)

    val r = Random(643895747384642L)
    val p: Int = (5 * BLOCK_SIZE) / 4
    val q: Int = (7 * BLOCK_SIZE) / 4
    matrix = createTestMatrix(r, p, q)
    checkUpperTriangular(QRDecomposition(matrix).r)

    matrix = createTestMatrix(r, p, q)
    checkUpperTriangular(QRDecomposition(matrix).r)
  }

  private fun checkUpperTriangular(m: RealMatrix) {
    m.walkInOptimizedOrder(object : DefaultRealMatrixPreservingVisitor() {
      override fun visit(row: Int, column: Int, value: Double) {
        if (column < row) {
          assertEquals(0.0, value, entryTolerance)
        }
      }
    })
  }

  /** test that H is trapezoidal  */
  @Test
  fun testHTrapezoidal() {
    var matrix: RealMatrix = mk.ndarray(testData3x3NonSingular)
    checkTrapezoidal(QRDecomposition(matrix).h)

    matrix = mk.ndarray(testData3x3Singular)
    checkTrapezoidal(QRDecomposition(matrix).h)

    matrix = mk.ndarray(testData3x4)
    checkTrapezoidal(QRDecomposition(matrix).h)

    matrix = mk.ndarray(testData4x3)
    checkTrapezoidal(QRDecomposition(matrix).h)

    val r = Random(643895747384642L)
    val p: Int = (5 * BLOCK_SIZE) / 4
    val q: Int = (7 * BLOCK_SIZE) / 4
    matrix = createTestMatrix(r, p, q)
    checkTrapezoidal(QRDecomposition(matrix).h)

    matrix = createTestMatrix(r, p, q)
    checkTrapezoidal(QRDecomposition(matrix).h)
  }

  private fun checkTrapezoidal(m: RealMatrix) {
    m.walkInOptimizedOrder(object : DefaultRealMatrixPreservingVisitor() {
      override fun visit(row: Int, column: Int, value: Double) {
        if (column > row) {
          assertEquals(0.0, value, entryTolerance)
        }
      }
    })
  }

  /** test matrices values  */
  @Test
  fun testMatricesValues() {
    val qr = QRDecomposition(mk.ndarray(testData3x3NonSingular))
    val qRef: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(-12.0 / 14.0, 69.0 / 175.0, -58.0 / 175.0),
        doubleArrayOf(-6.0 / 14.0, -158.0 / 175.0, 6.0 / 175.0),
        doubleArrayOf(4.0 / 14.0, -30.0 / 175.0, -165.0 / 175.0)
      )
    )
    val rRef: RealMatrix = mk.ndarray(
      arrayOf<DoubleArray>(
        doubleArrayOf(-14.0, -21.0, 14.0),
        doubleArrayOf(0.0, -175.0, 70.0),
        doubleArrayOf(0.0, 0.0, 35.0)
      )
    )
    val hRef: RealMatrix = mk.ndarray(
      arrayOf<DoubleArray>(
        doubleArrayOf(26.0 / 14.0, 0.0, 0.0),
        doubleArrayOf(6.0 / 14.0, 648.0 / 325.0, 0.0),
        doubleArrayOf(-4.0 / 14.0, 36.0 / 325.0, 2.0)
      )
    )

    // check values against known references
    val q: RealMatrix = qr.q
    assertEquals(0.0, q.minus(qRef).getNorm(), 1.0e-13)
    val qT: RealMatrix = qr.qT
    assertEquals(0.0, qT.minus(qRef.transpose()).getNorm(), 1.0e-13)
    val r: RealMatrix = qr.r
    assertEquals(0.0, r.minus(rRef).getNorm(), 1.0e-13)
    val h: RealMatrix = qr.h
    assertEquals(0.0, h.minus(hRef).getNorm(), 1.0e-13)

    // check the same cached instance is returned the second time
    assertTrue(q === qr.q)
    assertTrue(r === qr.r)
    assertTrue(h === qr.h)
  }

  @Test
  fun testNonInvertible() {
    val qr = QRDecomposition(mk.ndarray(testData3x3Singular))

    assertFailsWith(SingularMatrixException::class) { qr.solver.inverse }
  }

  private fun createTestMatrix(r: Random, rows: Int, columns: Int): RealMatrix {
    val m = mk.zeros<Double>(rows, columns)
    m.walkInOptimizedOrder(object : DefaultRealMatrixChangingVisitor() {
      override fun visit(row: Int, column: Int, value: Double): Double {
        return 2.0 * r.nextDouble() - 1.0
      }
    })
    return m
  }

  companion object {
    private const val entryTolerance = 10e-16

    private const val normTolerance = 10e-14
  }
}
