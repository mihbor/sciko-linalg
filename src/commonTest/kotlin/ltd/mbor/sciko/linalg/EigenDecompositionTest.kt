package ltd.mbor.sciko.linalg

import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.junit.Assert
import kotlin.math.sqrt
import kotlin.test.Test

class EigenDecompositionTest {

  @Test
  fun test() {

    mk.setEngine(KEEngineType)
    val A = mk.ndarray(mk[
      mk[-10.0                        , 5*sqrt(2.0)*(6 - 5*sqrt(3.0)), 5*sqrt(2.0)*(6 + 5*sqrt(3.0))],
      mk[5*sqrt(2.0)*(6 - 5*sqrt(3.0)), 5 - 20*sqrt(3.0)             , 115.0                        ],
      mk[5*sqrt(2.0)*(6 + 5*sqrt(3.0)), 115.0                        , 5 + 20*sqrt(3.0)             ],
    ])/16.0
    println(A)
    val eig = EigenDecomposition(A)
    println(eig.d)
    println(eig.v.transpose())
  }

  @Test
  fun testDimension1() {
    val matrix: RealMatrix = mk.ndarray(arrayOf(doubleArrayOf(1.5)))
    val ed = EigenDecomposition(matrix)
    Assert.assertEquals(1.5, ed.getRealEigenvalue(0), 1.0e-15)
  }

  @Test
  fun testDimension2() {
    val matrix: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(59.0, 12.0),
        doubleArrayOf(12.0, 66.0)
      )
    )
    val ed = EigenDecomposition(matrix)
    Assert.assertEquals(75.0, ed.getRealEigenvalue(0), 1.0e-15)
    Assert.assertEquals(50.0, ed.getRealEigenvalue(1), 1.0e-15)
  }

  @Test
  fun testDimension3() {
    val matrix: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(39632.0, -4824.0, -16560.0),
        doubleArrayOf(-4824.0, 8693.0, 7920.0),
        doubleArrayOf(-16560.0, 7920.0, 17300.0)
      )
    )
    val ed = EigenDecomposition(matrix)
    Assert.assertEquals(50000.0, ed.getRealEigenvalue(0), 3.0e-11)
    Assert.assertEquals(12500.0, ed.getRealEigenvalue(1), 3.0e-11)
    Assert.assertEquals(3125.0, ed.getRealEigenvalue(2), 3.0e-11)
  }

  @Test
  fun testDimension3MultipleRoot() {
    val matrix: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(5.0, 10.0, 15.0),
        doubleArrayOf(10.0, 20.0, 30.0),
        doubleArrayOf(15.0, 30.0, 45.0)
      )
    )
    val ed = EigenDecomposition(matrix)
    Assert.assertEquals(70.0, ed.getRealEigenvalue(0), 3.0e-11)
    Assert.assertEquals(0.0, ed.getRealEigenvalue(1), 3.0e-11)
    Assert.assertEquals(0.0, ed.getRealEigenvalue(2), 3.0e-11)
  }

  @Test
  fun testDimension4WithSplit() {
    val matrix: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(0.784, -0.288, 0.000, 0.000),
        doubleArrayOf(-0.288, 0.616, 0.000, 0.000),
        doubleArrayOf(0.000, 0.000, 0.164, -0.048),
        doubleArrayOf(0.000, 0.000, -0.048, 0.136)
      )
    )
    val ed = EigenDecomposition(matrix)
    Assert.assertEquals(1.0, ed.getRealEigenvalue(0), 1.0e-15)
    Assert.assertEquals(0.4, ed.getRealEigenvalue(1), 1.0e-15)
    Assert.assertEquals(0.2, ed.getRealEigenvalue(2), 1.0e-15)
    Assert.assertEquals(0.1, ed.getRealEigenvalue(3), 1.0e-15)
  }

  @Test
  fun testDimension4WithoutSplit() {
    val matrix: RealMatrix = mk.ndarray(
      arrayOf(
        doubleArrayOf(0.5608, -0.2016, 0.1152, -0.2976),
        doubleArrayOf(-0.2016, 0.4432, -0.2304, 0.1152),
        doubleArrayOf(0.1152, -0.2304, 0.3088, -0.1344),
        doubleArrayOf(-0.2976, 0.1152, -0.1344, 0.3872)
      )
    )
    val ed = EigenDecomposition(matrix)
    Assert.assertEquals(1.0, ed.getRealEigenvalue(0), 1.0e-15)
    Assert.assertEquals(0.4, ed.getRealEigenvalue(1), 1.0e-15)
    Assert.assertEquals(0.2, ed.getRealEigenvalue(2), 1.0e-15)
    Assert.assertEquals(0.1, ed.getRealEigenvalue(3), 1.0e-15)
  }

  // the following test triggered an ArrayIndexOutOfBoundsException in commons-math 2.0
  @Test
  fun testMath308() {
    mk.setEngine(KEEngineType)
    val mainTridiagonal = doubleArrayOf(
      22.330154644539597, 46.65485522478641, 17.393672330044705, 54.46687435351116, 80.17800767709437
    )
    val secondaryTridiagonal = doubleArrayOf(
      13.04450406501361, -5.977590941539671, 2.9040909856707517, 7.1570352792841225
    )
    // the reference values have been computed using routine DSTEMR
    // from the fortran library LAPACK version 3.2.1
    val refEigenValues = doubleArrayOf(
      82.044413207204002, 53.456697699894512, 52.536278520113882, 18.847969733754262, 14.138204224043099
    )
    val refEigenVectors = arrayOf<RealVector?>(
      mk.ndarray(doubleArrayOf(-0.000462690386766, -0.002118073109055, 0.011530080757413, 0.252322434584915, 0.967572088232592)),
      mk.ndarray(doubleArrayOf(0.314647769490148, 0.750806415553905, -0.167700312025760, -0.537092972407375, 0.143854968127780)),
      mk.ndarray(doubleArrayOf(0.222368839324646, 0.514921891363332, -0.021377019336614, 0.801196801016305, -0.207446991247740)),
      mk.ndarray(doubleArrayOf(-0.713933751051495, 0.190582113553930, -0.671410443368332, 0.056056055955050, -0.006541576993581)),
      mk.ndarray(doubleArrayOf(-0.584677060845929, 0.367177264979103, 0.721453187784497, -0.052971054621812, 0.005740715188257))
    )
    val decomposition = EigenDecomposition(mainTridiagonal, secondaryTridiagonal)
    val eigenValues = decomposition.getRealEigenvalues()
    for (i in refEigenValues.indices) {
      Assert.assertEquals(refEigenValues[i], eigenValues[i], 1.0e-5)
      Assert.assertEquals(0.0, (refEigenVectors[i]!! - decomposition.getEigenvector(i)).norm(), 2.0e-7)
    }
  }

  @Test
  fun testMathpbx02() {
    val mainTridiagonal = doubleArrayOf(
      7484.860960227216, 18405.28129035345, 13855.225609560746,
      10016.708722343366, 559.8117399576674, 6750.190788301587,
      71.21428769782159
    )
    val secondaryTridiagonal = doubleArrayOf(
      -4175.088570476366, 1975.7955858241994, 5193.178422374075,
      1995.286659169179, 75.34535882933804, -234.0808002076056
    )
    // the reference values have been computed using routine DSTEMR
    // from the fortran library LAPACK version 3.2.1
    val refEigenValues = doubleArrayOf(
      20654.744890306974412, 16828.208208485466457,
      6893.155912634994820, 6757.083016675340332,
      5887.799885688558788, 64.309089923240379,
      57.992628792736340
    )
    val refEigenVectors = arrayOf<RealVector?>(
      mk.ndarray(doubleArrayOf(-0.270356342026904, 0.852811091326997, 0.399639490702077, 0.198794657813990, 0.019739323307666, 0.000106983022327, -0.000001216636321)),
      mk.ndarray(doubleArrayOf(0.179995273578326, -0.402807848153042, 0.701870993525734, 0.555058211014888, 0.068079148898236, 0.000509139115227, -0.000007112235617)),
      mk.ndarray(doubleArrayOf(-0.399582721284727, -0.056629954519333, -0.514406488522827, 0.711168164518580, 0.225548081276367, 0.125943999652923, -0.004321507456014)),
      mk.ndarray(doubleArrayOf(0.058515721572821, 0.010200130057739, 0.063516274916536, -0.090696087449378, -0.017148420432597, 0.991318870265707, -0.034707338554096)),
      mk.ndarray(doubleArrayOf(0.855205995537564, 0.327134656629775, -0.265382397060548, 0.282690729026706, 0.105736068025572, -0.009138126622039, 0.000367751821196)),
      mk.ndarray(doubleArrayOf(-0.002913069901144, -0.005177515777101, 0.041906334478672, -0.109315918416258, 0.436192305456741, 0.026307315639535, 0.891797507436344)),
      mk.ndarray(doubleArrayOf(-0.005738311176435, -0.010207611670378, 0.082662420517928, -0.215733886094368, 0.861606487840411, -0.025478530652759, -0.451080697503958))
    )
    // the following line triggers the exception
    val decomposition: EigenDecomposition?
    decomposition = EigenDecomposition(mainTridiagonal, secondaryTridiagonal)
    val eigenValues = decomposition.getRealEigenvalues()
    for (i in refEigenValues.indices) {
      Assert.assertEquals(refEigenValues[i], eigenValues[i], 1.0e-3)
      if (refEigenVectors[i]!!.dot(decomposition.getEigenvector(i)) < 0) {
        Assert.assertEquals(0.0, refEigenVectors[i]!!.plus(decomposition.getEigenvector(i)).norm(), 1.0e-5)
      } else {
        Assert.assertEquals(0.0, refEigenVectors[i]!!.minus(decomposition.getEigenvector(i)).norm(), 1.0e-5)
      }
    }
  }

  @Test
  fun testMathpbx03() {
    val mainTridiagonal = doubleArrayOf(
      1809.0978259647177, 3395.4763425956166, 1832.1894584712693, 3804.364873592377,
      806.0482458637571, 2403.656427234185, 28.48691431556015
    )
    val secondaryTridiagonal = doubleArrayOf(
      -656.8932064545833, -469.30804108920734, -1021.7714889369421,
      -1152.540497328983, -939.9765163817368, -12.885877015422391
    )
    // the reference values have been computed using routine DSTEMR
    // from the fortran library LAPACK version 3.2.1
    val refEigenValues = doubleArrayOf(
      4603.121913685183245, 3691.195818048970978, 2743.442955402465032, 1657.596442107321764,
      1336.797819095331306, 30.129865209677519, 17.035352085224986
    )
    val refEigenVectors = arrayOf<RealVector?>(
      mk.ndarray(doubleArrayOf(-0.036249830202337, 0.154184732411519, -0.346016328392363, 0.867540105133093, -0.294483395433451, 0.125854235969548, -0.000354507444044)),
      mk.ndarray(doubleArrayOf(-0.318654191697157, 0.912992309960507, -0.129270874079777, -0.184150038178035, 0.096521712579439, -0.070468788536461, 0.000247918177736)),
      mk.ndarray(doubleArrayOf(-0.051394668681147, 0.073102235876933, 0.173502042943743, -0.188311980310942, -0.327158794289386, 0.905206581432676, -0.004296342252659)),
      mk.ndarray(doubleArrayOf(0.838150199198361, 0.193305209055716, -0.457341242126146, -0.166933875895419, 0.094512811358535, 0.119062381338757, -0.000941755685226)),
      mk.ndarray(doubleArrayOf(0.438071395458547, 0.314969169786246, 0.768480630802146, 0.227919171600705, -0.193317045298647, -0.170305467485594, 0.001677380536009)),
      mk.ndarray(doubleArrayOf(-0.003726503878741, -0.010091946369146, -0.067152015137611, -0.113798146542187, -0.313123000097908, -0.118940107954918, 0.932862311396062)),
      mk.ndarray(doubleArrayOf(0.009373003194332, 0.025570377559400, 0.170955836081348, 0.291954519805750, 0.807824267665706, 0.320108347088646, 0.360202112392266)),
    )
    // the following line triggers the exception
    val decomposition: EigenDecomposition?
    decomposition = EigenDecomposition(mainTridiagonal, secondaryTridiagonal)
    val eigenValues = decomposition.getRealEigenvalues()
    for (i in refEigenValues.indices) {
      Assert.assertEquals(refEigenValues[i], eigenValues[i], 1.0e-4)
      if (refEigenVectors[i]!!.dot(decomposition.getEigenvector(i)) < 0) {
        Assert.assertEquals(0.0, refEigenVectors[i]!!.plus(decomposition.getEigenvector(i)).norm(), 1.0e-5)
      } else {
        Assert.assertEquals(0.0, refEigenVectors[i]!!.minus(decomposition.getEigenvector(i)).norm(), 1.0e-5)
      }
    }
  }
}