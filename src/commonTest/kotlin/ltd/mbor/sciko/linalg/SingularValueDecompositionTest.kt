import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.MutableMultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.forEachMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

class SingularValueDecompositionTest {
  private val testSquare = arrayOf(
    doubleArrayOf(24.0 / 25.0, 43.0 / 25.0),
    doubleArrayOf(57.0 / 25.0, 24.0 / 25.0)
  )
  private val testNonSquare = arrayOf(
    doubleArrayOf(-540.0 / 625.0, 963.0 / 625.0, -216.0 / 625.0),
    doubleArrayOf(-1730.0 / 625.0, -744.0 / 625.0, 1008.0 / 625.0),
    doubleArrayOf(-720.0 / 625.0, 1284.0 / 625.0, -288.0 / 625.0),
    doubleArrayOf(-360.0 / 625.0, 192.0 / 625.0, 1756.0 / 625.0),
  )

  @Test
  fun testMoreRows() {
    val singularValues = doubleArrayOf(123.456, 2.3, 1.001, 0.999)
    val rows = singularValues.size + 2
    val columns = singularValues.size
    val r = Random(15338437322523L)
    val svd =
      SingularValueDecomposition(createTestMatrix(r, rows, columns, singularValues))
    val computedSV = svd.singularValues
    assertEquals(singularValues.size.toLong(), computedSV.size.toLong())
    for (i in singularValues.indices) {
      assertEquals(singularValues[i], computedSV[i], 1.0e-10)
    }
  }

  @Test
  fun testMoreColumns() {
    val singularValues = doubleArrayOf(123.456, 2.3, 1.001, 0.999)
    val rows = singularValues.size
    val columns = singularValues.size + 2
    val r = Random(732763225836210L)
    val svd =
      SingularValueDecomposition(createTestMatrix(r, rows, columns, singularValues))
    val computedSV = svd.singularValues
    assertEquals(singularValues.size.toLong(), computedSV.size.toLong())
    for (i in singularValues.indices) {
      assertEquals(singularValues[i], computedSV[i], 1.0e-10)
    }
  }

  /** test dimensions  */
  @Test
  fun testDimensions() {
    val matrix = mk.ndarray(testSquare)
    val m = matrix.rowDimension
    val n = matrix.columnDimension
    val svd = SingularValueDecomposition(matrix)
    assertEquals(m.toLong(), svd.u.rowDimension.toLong())
    assertEquals(m.toLong(), svd.u.columnDimension.toLong())
    assertEquals(m.toLong(), svd.s.columnDimension.toLong())
    assertEquals(n.toLong(), svd.s.columnDimension.toLong())
    assertEquals(n.toLong(), svd.v.rowDimension.toLong())
    assertEquals(n.toLong(), svd.v.columnDimension.toLong())
  }

  /** Test based on a dimension 4 Hadamard matrix.  */
  @Test
  fun testHadamard() {
    val matrix: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(15.0 / 2.0, 5.0 / 2.0, 9.0 / 2.0, 3.0 / 2.0),
        doubleArrayOf(5.0 / 2.0, 15.0 / 2.0, 3.0 / 2.0, 9.0 / 2.0),
        doubleArrayOf(9.0 / 2.0, 3.0 / 2.0, 15.0 / 2.0, 5.0 / 2.0),
        doubleArrayOf(3.0 / 2.0, 9.0 / 2.0, 5.0 / 2.0, 15.0 / 2.0)
      )
    )
    val svd = SingularValueDecomposition(matrix)
    assertEquals(16.0, svd.singularValues[0], 1.0e-14)
    assertEquals(8.0, svd.singularValues[1], 1.0e-14)
    assertEquals(4.0, svd.singularValues[2], 1.0e-14)
    assertEquals(2.0, svd.singularValues[3], 1.0e-14)
    val fullCovariance: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(85.0 / 1024, -51.0 / 1024, -75.0 / 1024, 45.0 / 1024),
        doubleArrayOf(-51.0 / 1024, 85.0 / 1024, 45.0 / 1024, -75.0 / 1024),
        doubleArrayOf(-75.0 / 1024, 45.0 / 1024, 85.0 / 1024, -51.0 / 1024),
        doubleArrayOf(45.0 / 1024, -75.0 / 1024, -51.0 / 1024, 85.0 / 1024)
      )
    )
    assertEquals(
      0.0,
      fullCovariance.minus(svd.getCovariance(0.0)).norm,
      1.0e-14
    )
    val halfCovariance: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(5.0 / 1024, -3.0 / 1024, 5.0 / 1024, -3.0 / 1024),
        doubleArrayOf(-3.0 / 1024, 5.0 / 1024, -3.0 / 1024, 5.0 / 1024),
        doubleArrayOf(5.0 / 1024, -3.0 / 1024, 5.0 / 1024, -3.0 / 1024),
        doubleArrayOf(-3.0 / 1024, 5.0 / 1024, -3.0 / 1024, 5.0 / 1024)
      )
    )
    assertEquals(
      0.0,
      halfCovariance.minus(svd.getCovariance(6.0)).norm,
      1.0e-14
    )
  }

  /** test A = USVt  */
  @Test
  fun testAEqualUSVt() {
    checkAEqualUSVt(mk.ndarray(testSquare))
    checkAEqualUSVt(mk.ndarray(testNonSquare))
    checkAEqualUSVt(mk.ndarray(testNonSquare).transpose())
  }

  fun checkAEqualUSVt(matrix: RealMatrix) {
    val svd = SingularValueDecomposition(matrix)
    val u = svd.u
    val s = svd.s
    val v = svd.v
    val norm = u.dot(s).dot(v.transpose()).minus(matrix).norm
    assertEquals(0.0, norm, normTolerance)
  }

  /** test that U is orthogonal  */
  @Test
  fun testUOrthogonal() {
    checkOrthogonal(SingularValueDecomposition(mk.ndarray(testSquare)).u)
    checkOrthogonal(SingularValueDecomposition(mk.ndarray(testNonSquare)).u)
    checkOrthogonal(SingularValueDecomposition(mk.ndarray(testNonSquare).transpose()).u)
  }

  /** test that V is orthogonal  */
  @Test
  fun testVOrthogonal() {
    checkOrthogonal(SingularValueDecomposition(mk.ndarray(testSquare)).v)
    checkOrthogonal(SingularValueDecomposition(mk.ndarray(testNonSquare)).v)
    checkOrthogonal(SingularValueDecomposition(mk.ndarray(testNonSquare).transpose()).v)
  }

  fun checkOrthogonal(m: RealMatrix) {
    val mTm = m.transpose().dot(m)
    val id = mk.identity<Double>(mTm.rowDimension)
    assertEquals(0.0, mTm.minus(id).norm, normTolerance)
  }

  /** test matrices values  */ // This test is useless since whereas the columns of U and V are linked
  // together, the actual triplet (U,S,V) is not uniquely defined.
  fun testMatricesValues1() {
    val svd =
      SingularValueDecomposition(mk.ndarray(testSquare))
    val uRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(3.0 / 5.0, -4.0 / 5.0),
        doubleArrayOf(4.0 / 5.0, 3.0 / 5.0)
      )
    )
    val sRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(3.0, 0.0),
        doubleArrayOf(0.0, 1.0)
      )
    )
    val vRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(4.0 / 5.0, 3.0 / 5.0),
        doubleArrayOf(3.0 / 5.0, -4.0 / 5.0)
      )
    )
    // check values against known references
    val u = svd.u
    assertEquals(0.0, u.minus(uRef).norm, normTolerance)
    val s = svd.s
    assertEquals(0.0, s.minus(sRef).norm, normTolerance)
    val v = svd.v
    assertEquals(0.0, v.minus(vRef).norm, normTolerance)
    // check the same cached instance is returned the second time
    assertTrue(u === svd.u)
    assertTrue(s === svd.s)
    assertTrue(v === svd.v)
  }

  /** test matrices values  */ // This test is useless since whereas the columns of U and V are linked
  // together, the actual triplet (U,S,V) is not uniquely defined.
  fun useless_testMatricesValues2() {
    val uRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(0.0 / 5.0, 3.0 / 5.0, 0.0 / 5.0),
        doubleArrayOf(-4.0 / 5.0, 0.0 / 5.0, -3.0 / 5.0),
        doubleArrayOf(0.0 / 5.0, 4.0 / 5.0, 0.0 / 5.0),
        doubleArrayOf(-3.0 / 5.0, 0.0 / 5.0, 4.0 / 5.0)
      )
    )
    val sRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(4.0, 0.0, 0.0),
        doubleArrayOf(0.0, 3.0, 0.0),
        doubleArrayOf(0.0, 0.0, 2.0)
      )
    )
    val vRef = mk.ndarray(
      arrayOf(
        doubleArrayOf(80.0 / 125.0, -60.0 / 125.0, 75.0 / 125.0),
        doubleArrayOf(24.0 / 125.0, 107.0 / 125.0, 60.0 / 125.0),
        doubleArrayOf(-93.0 / 125.0, -24.0 / 125.0, 80.0 / 125.0)
      )
    )
    // check values against known references
    val svd =
      SingularValueDecomposition(mk.ndarray(testNonSquare))
    val u = svd.u
    assertEquals(0.0, u.minus(uRef).norm, normTolerance)
    val s = svd.s
    assertEquals(0.0, s.minus(sRef).norm, normTolerance)
    val v = svd.v
    assertEquals(0.0, v.minus(vRef).norm, normTolerance)
    // check the same cached instance is returned the second time
    assertTrue(u === svd.u)
    assertTrue(s === svd.s)
    assertTrue(v === svd.v)
  }

  /** test MATH-465  */
  @Test
  fun testRank() {
    val d = arrayOf(doubleArrayOf(1.0, 1.0, 1.0), doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(1.0, 2.0, 3.0))
    val m: RealMatrix = mk.ndarray(d)
    val svd = SingularValueDecomposition(m)
    assertEquals(2, svd.rank.toLong())
  }

  /** test MATH-583  */
  @Test
  fun testStability1() {
    val m = mk.zeros<Double>(201, 201)
    loadRealMatrix(m, "matrix1.csv")
    try {
      SingularValueDecomposition(m)
    } catch (e: Exception) {
      fail("Exception whilst constructing SVD")
    }
  }

  /** test MATH-327  */
  @Test
  fun testStability2() {
    val m = mk.zeros<Double>(7, 168)
    loadRealMatrix(m, "matrix2.csv")
    try {
      SingularValueDecomposition(m)
    } catch (e: Throwable) {
      fail("Exception whilst constructing SVD")
    }
  }

  private fun loadRealMatrix(m: MutableMultiArray<Double, D2>, resourceName: String) {
    val `in` = checkNotNull(javaClass.getResourceAsStream(resourceName))
    val br = `in`.reader().buffered()
    var strLine: String
    var row = 0
    while ((br.readLine().also { strLine = it }) != null) {
      if (!strLine.startsWith("#")) {
        var col = 0
        for (entry in strLine.split(",".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()) {
          m.set(row, col++, entry.toDouble())
        }
        row++
      }
    }
    `in`.close()
  }

  /** test condition number  */
  @Test
  fun testConditionNumber() {
    val svd =
      SingularValueDecomposition(mk.ndarray(testSquare))
    // replace 1.0e-15 with 1.5e-15
    assertEquals(3.0, svd.conditionNumber, 1.5e-15)
  }

  @Test
  fun testInverseConditionNumber() {
    val svd =
      SingularValueDecomposition(mk.ndarray(testSquare))
    assertEquals(1.0 / 3.0, svd.inverseConditionNumber, 1.5e-15)
  }

  private fun createTestMatrix(
    r: Random, rows: Int, columns: Int,
    singularValues: DoubleArray
  ): RealMatrix {
    val u: RealMatrix = createOrthogonalMatrix(r, rows)
    val d = mk.zeros<Double>(rows, columns)
    d.setSubMatrix(mk.diagonal(singularValues.toList()), 0, 0)
    val v: RealMatrix = createOrthogonalMatrix(r, columns)
    return u.dot(d).dot(v)
  }

  @Test
  fun testIssue947() {
    val nans = arrayOf(
      doubleArrayOf(Double.NaN, Double.NaN),
      doubleArrayOf(Double.NaN, Double.NaN)
    )
    val m: RealMatrix = mk.ndarray(nans)
    val svd = SingularValueDecomposition(m)
    assertTrue(java.lang.Double.isNaN(svd.singularValues[0]))
    assertTrue(java.lang.Double.isNaN(svd.singularValues[1]))
  }

  companion object {
    private const val normTolerance = 10e-14
  }
}

private fun MutableMultiArray<Double, D2>.setSubMatrix(other: MultiArray<Double, D2>, iStart: Int, jStart: Int) {
  other.forEachMultiIndexed{ i, it ->
    this[i[0] + iStart, i[1] + jStart] = it
  }
}

fun createOrthogonalMatrix(r: Random, size: Int): RealMatrix {
  val data = Array(size) { DoubleArray(size) }
  for (i in 0..<size) {
    val dataI = data[i]
    var norm2 = 0.0
    do {
      // generate randomly row I
      for (j in 0..<size) {
        dataI[j] = 2 * r.nextDouble() - 1
      }
      // project the row in the subspace orthogonal to previous rows
      for (k in 0..<i) {
        val dataK = data[k]
        var dotProduct = 0.0
        for (j in 0..<size) {
          dotProduct += dataI[j] * dataK[j]
        }
        for (j in 0..<size) {
          dataI[j] -= dotProduct * dataK[j]
        }
      }
      // normalize the row
      norm2 = 0.0
      for (dataIJ in dataI) {
        norm2 += dataIJ * dataIJ
      }
      val inv = 1.0 / FastMath.sqrt(norm2)
      for (j in 0..<size) {
        dataI[j] *= inv
      }
    } while (norm2 * size < 0.01)
  }
  return mk.ndarray(data)
}