plugins {
  kotlin("multiplatform") version "2.0.21"
  id("maven-publish")
}

group = "ltd.mbor.sciko"
version = "0.1-SNAPSHOT"

repositories {
  mavenCentral()
}

kotlin {
  sourceSets {
    val commonMain by getting {
      dependencies {
        api("org.jetbrains.kotlinx:multik-core:0.2.3")
        implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
      }
    }
    val commonTest by getting {
      dependencies {
        implementation(kotlin("test"))
      }
    }
  }
  jvm {
    testRuns["test"].executionTask.configure {
      useJUnitPlatform()
    }
  }
  jvmToolchain(21)
}
