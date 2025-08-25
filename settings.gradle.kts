rootProject.name = "sciko-linalg"
pluginManagement {
  repositories {
    gradlePluginPortal()
    mavenCentral()
    // Add Google repository for Android plugin
    maven("https://maven.google.com/") {
      content {
        includeGroupByRegex("com\\.android.*")
        includeGroupByRegex("com\\.google.*")
        includeGroupByRegex("androidx.*")
      }
    }
  }
}
plugins {
  id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
