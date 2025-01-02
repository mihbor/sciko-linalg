package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.test.*

class LUDecompositionTest {
  private val testData = arrayOf(
    doubleArrayOf(1.0, 2.0, 3.0),
    doubleArrayOf(2.0, 5.0, 3.0),
    doubleArrayOf(1.0, 0.0, 8.0)
  )
  private val testDataMinus = arrayOf(
    doubleArrayOf(-1.0, -2.0, -3.0),
    doubleArrayOf(-2.0, -5.0, -3.0),
    doubleArrayOf(-1.0, 0.0, -8.0)
  )
  private val luData = arrayOf(
    doubleArrayOf(2.0, 3.0, 3.0),
    doubleArrayOf(0.0, 5.0, 7.0),
    doubleArrayOf(6.0, 9.0, 8.0)
  )

  // singular matrices
  private val singular = arrayOf(
    doubleArrayOf(2.0, 3.0),
    doubleArrayOf(2.0, 3.0)
  )
  private val bigSingular = arrayOf(
    doubleArrayOf(1.0, 2.0, 3.0, 4.0),
    doubleArrayOf(2.0, 5.0, 3.0, 4.0),
    doubleArrayOf(7.0, 3.0, 256.0, 1930.0),
    doubleArrayOf(3.0, 7.0, 6.0, 8.0)
  ) // 4th row = 1st + 2nd

  /** test dimensions  */
  @Test
  fun testDimensions() {
    val matrix = mk.ndarray(testData)
    val LU = LUDecomposition(matrix)
    assertEquals(testData.size.toLong(), LU.l!!.rowDimension.toLong())
    assertEquals(testData.size.toLong(), LU.l!!.columnDimension.toLong())
    assertEquals(testData.size.toLong(), LU.u!!.rowDimension.toLong())
    assertEquals(testData.size.toLong(), LU.u!!.columnDimension.toLong())
    assertEquals(testData.size.toLong(), LU.p!!.rowDimension.toLong())
    assertEquals(testData.size.toLong(), LU.p!!.columnDimension.toLong())
  }

  /** test non-square matrix  */
  @Test
  fun testNonSquare() {
    try {
      LUDecomposition(mk.ndarray(Array(3) { DoubleArray(2) }))
      fail("Expecting NonSquareMatrixException")
    } catch (ime: NonSquareMatrixException) {
      // expected behavior
    }
  }

  /** test PA = LU  */
  @Test
  fun testPAEqualLU() {
    var matrix = mk.ndarray(testData)
    var lu = LUDecomposition(matrix)
    var l = lu.l!!
    var u = lu.u!!
    var p = lu.p!!
    var norm = l.dot(u).minus(p.dot(matrix)).norm
    assertEquals(0.0, norm, normTolerance)
    matrix = mk.ndarray(testDataMinus)
    lu = LUDecomposition(matrix)
    l = lu.l!!
    u = lu.u!!
    p = lu.p!!
    norm = l.dot(u).minus(p.dot(matrix)).norm
    assertEquals(0.0, norm, normTolerance)
    matrix = mk.identity(17)
    lu = LUDecomposition(matrix)
    l = lu.l!!
    u = lu.u!!
    p = lu.p!!
    norm = l.dot(u).minus(p.dot(matrix)).norm
    assertEquals(0.0, norm, normTolerance)
    matrix = mk.ndarray(singular)
    lu = LUDecomposition(matrix)
    assertFalse(lu.solver.isNonSingular)
    assertNull(lu.l)
    assertNull(lu.u)
    assertNull(lu.p)
    matrix = mk.ndarray(bigSingular)
    lu = LUDecomposition(matrix)
    assertFalse(lu.solver.isNonSingular)
    assertNull(lu.l)
    assertNull(lu.u)
    assertNull(lu.p)
  }

  /** test that L is lower triangular with unit diagonal  */
  @Test
  fun testLLowerTriangular() {
    val matrix = mk.ndarray(testData)
    val l = LUDecomposition(matrix).l!!
    for (i in 0..<l.rowDimension) {
      assertEquals(l.get(i, i), 1.0, entryTolerance)
      for (j in i + 1..<l.columnDimension) {
        assertEquals(l.get(i, j), 0.0, entryTolerance)
      }
    }
  }

  /** test that U is upper triangular  */
  @Test
  fun testUUpperTriangular() {
    val matrix = mk.ndarray(testData)
    val u = LUDecomposition(matrix).u!!
    for (i in 0..<u.rowDimension) {
      for (j in 0..<i) {
        assertEquals(u[i, j], 0.0, entryTolerance)
      }
    }
  }

  /** test that P is a permutation matrix  */
  @Test
  fun testPPermutation() {
    val matrix = mk.ndarray(testData)
    val p = LUDecomposition(matrix).p!!
    val ppT = p.dot(p.transpose())
    val id = mk.identity<Double>(p.rowDimension)
    assertEquals(0.0, ppT.minus(id).norm, normTolerance)
    for (i in 0..<p.rowDimension) {
      var zeroCount = 0
      var oneCount = 0
      var otherCount = 0
      for (j in 0..<p.columnDimension) {
        val e = p[i, j]
        if (e == 0.0) {
          ++zeroCount
        } else if (e == 1.0) {
          ++oneCount
        } else {
          ++otherCount
        }
      }
      assertEquals((p.columnDimension - 1).toLong(), zeroCount.toLong())
      assertEquals(1, oneCount.toLong())
      assertEquals(0, otherCount.toLong())
    }
    for (j in 0..<p.columnDimension) {
      var zeroCount = 0
      var oneCount = 0
      var otherCount = 0
      for (i in 0..<p.rowDimension) {
        val e = p[i, j]
        if (e == 0.0) {
          ++zeroCount
        } else if (e == 1.0) {
          ++oneCount
        } else {
          ++otherCount
        }
      }
      assertEquals((p.rowDimension - 1).toLong(), zeroCount.toLong())
      assertEquals(1, oneCount.toLong())
      assertEquals(0, otherCount.toLong())
    }
  }

  /** test singular  */
  @Test
  fun testSingular() {
    var lu =
      LUDecomposition(mk.ndarray(testData))
    assertTrue(lu.solver.isNonSingular)
    lu = LUDecomposition(mk.ndarray(singular))
    assertFalse(lu.solver.isNonSingular)
    lu = LUDecomposition(mk.ndarray(bigSingular))
    assertFalse(lu.solver.isNonSingular)
  }

  /** test matrices values  */
  @Test
  fun testMatricesValues1() {
    val lu =
      LUDecomposition(mk.ndarray(testData))
    val lRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0),
        doubleArrayOf(0.5, 1.0, 0.0),
        doubleArrayOf(0.5, 0.2, 1.0)
      )
    )
    val uRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(2.0, 5.0, 3.0),
        doubleArrayOf(0.0, -2.5, 6.5),
        doubleArrayOf(0.0, 0.0, 0.2)
      )
    )
    val pRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0),
        doubleArrayOf(1.0, 0.0, 0.0)
      )
    )
    val pivotRef = intArrayOf(1, 2, 0)
    // check values against known references
    val l = lu.l!!
    assertEquals(0.0, l.minus(lRef).norm, 1.0e-13)
    val u = lu.u!!
    assertEquals(0.0, u.minus(uRef).norm, 1.0e-13)
    val p = lu.p!!
    assertEquals(0.0, p.minus(pRef).norm, 1.0e-13)
    val pivot = lu.pivot
    for (i in pivotRef.indices) {
      assertEquals(pivotRef[i].toLong(), pivot[i].toLong())
    }
    // check the same cached instance is returned the second time
    assertTrue(l === lu.l)
    assertTrue(u === lu.u)
    assertTrue(p === lu.p)
  }

  /** test matrices values  */
  @Test
  fun testMatricesValues2() {
    val lu =
      LUDecomposition(mk.ndarray(luData))
    val lRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(1.0 / 3.0, 0.0, 1.0)
      )
    )
    val uRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(6.0, 9.0, 8.0),
        doubleArrayOf(0.0, 5.0, 7.0),
        doubleArrayOf(0.0, 0.0, 1.0 / 3.0)
      )
    )
    val pRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(0.0, 0.0, 1.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(1.0, 0.0, 0.0)
      )
    )
    val pivotRef = intArrayOf(2, 1, 0)
    // check values against known references
    val l = lu.l!!
    assertEquals(0.0, l.minus(lRef).norm, 1.0e-13)
    val u = lu.u!!
    assertEquals(0.0, u.minus(uRef).norm, 1.0e-13)
    val p = lu.p!!
    assertEquals(0.0, p.minus(pRef).norm, 1.0e-13)
    val pivot = lu.pivot
    for (i in pivotRef.indices) {
      assertEquals(pivotRef[i].toLong(), pivot[i].toLong())
    }
    // check the same cached instance is returned the second time
    assertTrue(l === lu.l)
    assertTrue(u === lu.u)
    assertTrue(p === lu.p)
  }

  companion object {
    private const val entryTolerance = 10e-16
    private const val normTolerance = 10e-14
  }
}