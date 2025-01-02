package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray

typealias FastMath = Math
typealias RealMatrix = MultiArray<Double, D2>
typealias RealVector = MultiArray<Double, D1>

val RealVector.dimension get() = shape[0]
val RealMatrix.rowDimension get() = shape[0]
val RealMatrix.columnDimension get() = shape[1]
fun RealMatrix.getData() = toArray()
fun RealVector.toArray() = toDoubleArray()

val RealMatrix.isSquare: Boolean
  get() = shape[0] == shape[1]

class NonSquareMatrixException(val rowDimension: Int, val columnDimension: Int):
  RuntimeException("Matrix not square $rowDimension != $columnDimension")

val RealMatrix.norm get() = mk.linalg.norm(this)

open class DefaultRealMatrixPreservingVisitor {
  /** {@inheritDoc}  */
  fun start(
    rows: Int, columns: Int,
    startRow: Int, endRow: Int, startColumn: Int, endColumn: Int
  ) {
  }

  /** {@inheritDoc}  */
  open fun visit(row: Int, column: Int, value: Double) {}

  /** {@inheritDoc}  */
  fun end(): Double {
    return 0.0
  }
}

open class DefaultRealMatrixChangingVisitor {
  /** {@inheritDoc}  */
  fun start(
    rows: Int, columns: Int,
    startRow: Int, endRow: Int, startColumn: Int, endColumn: Int
  ) {
  }

  /** {@inheritDoc}  */
  open fun visit(row: Int, column: Int, value: Double): Double {
    return value
  }

  /** {@inheritDoc}  */
  fun end(): Double {
    return 0.0
  }
}

fun MutableMultiArray<Double, D2>.walkInOptimizedOrder(visitor: DefaultRealMatrixChangingVisitor) = walkInRowOrder(visitor)

fun MutableMultiArray<Double, D2>.walkInRowOrder(visitor: DefaultRealMatrixChangingVisitor): Double {
  val rows: Int = rowDimension
  val columns: Int = columnDimension
  visitor.start(rows, columns, 0, rows - 1, 0, columns - 1)
  for (row in 0 until rows) {
    for (column in 0 until columns) {
      val oldValue: Double = this[row, column]
      val newValue: Double = visitor.visit(row, column, oldValue)
      this[row, column] = newValue
    }
  }
  return visitor.end()
}

fun RealMatrix.walkInOptimizedOrder(visitor: DefaultRealMatrixPreservingVisitor) = walkInRowOrder(visitor)

fun RealMatrix.walkInRowOrder(visitor: DefaultRealMatrixPreservingVisitor): Double {
  val rows: Int = rowDimension
  val columns: Int = columnDimension
  visitor.start(rows, columns, 0, rows - 1, 0, columns - 1)
  for (row in 0 until rows) {
    for (column in 0 until columns) {
      visitor.visit(row, column, this[row, column])
    }
  }
  return visitor.end()
}

fun RealMatrix.walkInOptimizedOrder(
  visitor: DefaultRealMatrixPreservingVisitor,
  startRow: Int, endRow: Int,
  startColumn: Int,
  endColumn: Int
): Double {
  return walkInRowOrder(visitor, startRow, endRow, startColumn, endColumn)
}

fun RealMatrix.walkInRowOrder(
  visitor: DefaultRealMatrixPreservingVisitor,
  startRow: Int, endRow: Int,
  startColumn: Int, endColumn: Int
): Double {
//  MatrixUtils.checkSubMatrixIndex(this, startRow, endRow, startColumn, endColumn)
  visitor.start(
    rowDimension, columnDimension,
    startRow, endRow, startColumn, endColumn
  )
  for (row in startRow..endRow) {
    for (column in startColumn..endColumn) {
      visitor.visit(row, column, get(row, column))
    }
  }
  return visitor.end()
}
