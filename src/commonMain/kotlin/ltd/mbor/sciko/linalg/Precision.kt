package ltd.mbor.sciko.linalg

class Precision {
  companion object {
    /** Exponent offset in IEEE754 representation. */
    val EXPONENT_OFFSET = 1023L
    /*
     * This was previously expressed as = 0x1.0p-1022;
     * However, OpenJDK (Sparc Solaris) cannot handle such small
     * constants: MATH-721
     */
    val SAFE_MIN = java.lang.Double.longBitsToDouble((EXPONENT_OFFSET - 1022L) shl 52)
    /*
     *  This was previously expressed as = 0x1.0p-53;
     *  However, OpenJDK (Sparc Solaris) cannot handle such small
     *  constants: MATH-721
     */
    val EPSILON = Double.fromBits((EXPONENT_OFFSET - 53L) shl 52)
    /** Positive zero bits.  */
    private val POSITIVE_ZERO_DOUBLE_BITS: Long = java.lang.Double.doubleToRawLongBits(+0.0)
    /** Negative zero bits.  */
    private val NEGATIVE_ZERO_DOUBLE_BITS: Long = java.lang.Double.doubleToRawLongBits(-0.0)

    /**
     * Returns `true` if there is no double value strictly between the
     * arguments or the difference between them is within the range of allowed
     * error (inclusive).
     *
     * @param x First value.
     * @param y Second value.
     * @param eps Amount of allowed absolute error.
     * @return `true` if the values are two adjacent floating point
     * numbers or they are within range of each other.
     */
    fun equals(x: Double, y: Double, eps: Double): Boolean {
      return equals(x, y, 1) || FastMath.abs(y - x) <= eps
    }

    /**
     * Returns true iff they are equal as defined by
     * [equals(x, y, 1)][.equals].
     *
     * @param x first value
     * @param y second value
     * @return `true` if the values are equal.
     */
    fun equals(x: Double, y: Double): Boolean {
      return equals(x, y, 1)
    }

    /**
     * Returns true if both arguments are equal or within the range of allowed
     * error (inclusive).
     *
     *
     * Two float numbers are considered equal if there are `(maxUlps - 1)`
     * (or fewer) floating point numbers between them, i.e. two adjacent
     * floating point numbers are considered equal.
     *
     *
     *
     * Adapted from [
 * Bruce Dawson](http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
     *
     *
     * @param x first value
     * @param y second value
     * @param maxUlps `(maxUlps - 1)` is the number of floating point
     * values between `x` and `y`.
     * @return `true` if there are fewer than `maxUlps` floating
     * point values between `x` and `y`.
     */
    fun equals(x: Double, y: Double, maxUlps: Int): Boolean {
      val xInt = java.lang.Double.doubleToRawLongBits(x)
      val yInt = java.lang.Double.doubleToRawLongBits(y)
      val isEqual: Boolean
      if ((xInt >= 0) == (yInt >= 0)) {
        // number have same sign, there is no risk of overflow
        isEqual = FastMath.abs(xInt - yInt) <= maxUlps
      } else {
        // number have opposite signs, take care of overflow
        val deltaPlus: Long
        val deltaMinus: Long
        if (xInt < yInt) {
          deltaPlus = yInt - POSITIVE_ZERO_DOUBLE_BITS
          deltaMinus = xInt - NEGATIVE_ZERO_DOUBLE_BITS
        } else {
          deltaPlus = xInt - POSITIVE_ZERO_DOUBLE_BITS
          deltaMinus = yInt - NEGATIVE_ZERO_DOUBLE_BITS
        }
        if (deltaPlus > maxUlps) {
          isEqual = false
        } else {
          isEqual = deltaMinus <= (maxUlps - deltaPlus)
        }
      }
      return isEqual && !java.lang.Double.isNaN(x) && !java.lang.Double.isNaN(y)
    }

    /**
     * Compares two numbers given some amount of allowed error.
     *
     * @param x the first number
     * @param y the second number
     * @param eps the amount of error to allow when checking for equality
     * @return  * 0 if  [equals(x, y, eps)][.equals]
     *  * &lt; 0 if ![equals(x, y, eps)][.equals] &amp;&amp; x &lt; y
     *  * > 0 if ![equals(x, y, eps)][.equals] &amp;&amp; x > y
     */
    fun compareTo(x: Double, y: Double, eps: Double): Int {
      if (equals(x, y, eps)) {
        return 0
      } else if (x < y) {
        return -1
      }
      return 1
    }
  }
}