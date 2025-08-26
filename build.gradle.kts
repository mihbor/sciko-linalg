import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension

plugins {
  kotlin("multiplatform") version "2.2.10"
  id("com.android.library") version "8.11.1"
  id("org.jetbrains.kotlin.android") version "2.2.10" apply false
  id("maven-publish")
}

group = "ltd.mbor.sciko"
version = "0.1-SNAPSHOT"

repositories {
  mavenCentral()
  google()
}

kotlin {
  jvm()
  jvmToolchain(21)
  js(IR) {
    browser()
    binaries.executable()
  }
  androidTarget {
    publishLibraryVariants("release", "debug")
  }
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
    val jvmTest by getting {
      dependencies {
        implementation(kotlin("test"))
      }
    }
  }
}

android {
  compileSdk = 36
  namespace = "ltd.mbor.sciko.linalg"
  defaultConfig {
    minSdk = 21
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
  }
}

publishing {
  repositories {
    maven {
      name = "GitHubPackages"
      url = uri("https://maven.pkg.github.com/mihbor/sciko-linalg")
      credentials {
        username = project.findProperty("gpr.user") as String? ?: System.getenv("USERNAME")
        password = project.findProperty("gpr.key") as String? ?: System.getenv("TOKEN")
      }
    }
  }
}

rootProject.plugins.withType(YarnPlugin::class.java) {
  rootProject.the<YarnRootExtension>().yarnLockAutoReplace = true
}
