import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension

plugins {
  kotlin("multiplatform") version "2.2.10"
  id("maven-publish")
}

group = "ltd.mbor.sciko"
version = "0.1-SNAPSHOT"

repositories {
  mavenCentral()
}

kotlin {
  jvm() {
    compilations.all {
      compileTaskProvider.configure {
        compilerOptions {
          jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_1_8) // Android compatibility
        }
      }
    }
  }
  jvmToolchain(21)
  js(IR) {
    browser()
    binaries.executable()
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
  
  publications.withType<MavenPublication> {
    // Add Android compatibility metadata to JVM publication
    if (name == "jvm") {
      pom {
        properties.put("android.compatible", "true")
        properties.put("jvm.target", "1.8")
      }
    }
  }
  
  // Create a dedicated Android publication based on JVM artifacts
  publications.register<MavenPublication>("android") {
    groupId = project.group.toString()
    artifactId = "${project.name}-android"
    version = project.version.toString()
    
    pom {
      name.set("${project.name}-android")
      description.set("Sciko Linear Algebra - Android Compatible Library")
      properties.put("android.compatible", "true")
      properties.put("target.platform", "android")
    }
    
    // Add artifacts after they are created
    afterEvaluate {
      artifact(tasks.getByName("jvmJar"))
      artifact(tasks.getByName("jvmSourcesJar"))
    }
  }
}

rootProject.plugins.withType(YarnPlugin::class.java) {
  rootProject.the<YarnRootExtension>().yarnLockAutoReplace = true
}
