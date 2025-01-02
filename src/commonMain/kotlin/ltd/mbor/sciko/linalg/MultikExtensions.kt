import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * Returns a diagonal array.
 *
 * @param elements the elements along the main diagonal.
 * @return [D2Array].
 * @sample samples.NDArrayTest.diagonal
 */
inline fun <reified T : Any> Multik.diagonal(elements: List<T>): D2Array<T> {
  val dtype = DataType.ofKClass(T::class)
  return diagonal(elements = elements, dtype = dtype)
}

/**
 * Returns a diagonal array.
 *
 * Note: Generic type of elements must match [dtype].
 *
 * @param dtype array type.
 * @param elements the elements on the main diagonal.
 * @return [D2Array]
 * @sample samples.NDArrayTest.diagonalWithDtype
 */
fun <T> Multik.diagonal(elements: List<T>, dtype: DataType): D2Array<T> {
  val n = elements.size
  val shape = intArrayOf(n, n)
  val ret = D2Array(initMemoryView<T>(n * n, dtype), shape = shape, dim = D2)
  for (i in 0 until n) {
    ret[i, i] = elements[i]
  }
  return ret
}

fun LinAlg.det(M: MultiArray<Double, D2>) = LUDecomposition(M).determinant

fun LinAlg.svd(m: MultiArray<Double, D2>): Triple<MultiArray<Double, D2>, MultiArray<Double, D2>, MultiArray<Double, D2>> {
  val decomposition = SingularValueDecomposition(m)
  return Triple(decomposition.u, decomposition.s, decomposition.vT)
}

fun LinAlg.pinv(m: MultiArray<Double, D2>): MultiArray<Double, D2> {
  val decomposition = SingularValueDecomposition(m)
  return decomposition.solver.inverse
}

fun LinAlg.leftInv(m: MultiArray<Double, D2>): MultiArray<Double, D2> {
  return inv(m.transpose() dot m) dot m.transpose()
}

fun LinAlg.rightInv(m: MultiArray<Double, D2>): MultiArray<Double, D2> {
  return m.transpose() dot inv(m dot m.transpose())
}
