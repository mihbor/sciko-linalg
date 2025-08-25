# Sciko Linear Algebra

A Kotlin Multiplatform library for linear algebra operations.

## Features

- Eigen decomposition for real matrices
- LU decomposition
- QR decomposition
- Schur transformation
- High precision floating point utilities
- Support for both JVM and JavaScript platforms

## Usage

### Gradle

Add the following to your `build.gradle.kts`:

```kotlin
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

dependencies {
    implementation("ltd.mbor.sciko:sciko-linalg:0.1-SNAPSHOT")
}
```

### Authentication

To use packages from GitHub Packages, you need to authenticate:

1. **Using environment variables:**
   ```bash
   export USERNAME=your-github-username
   export TOKEN=your-github-personal-access-token
   ```

2. **Using gradle.properties:**
   ```properties
   gpr.user=your-github-username
   gpr.key=your-github-personal-access-token
   ```

The personal access token needs the `read:packages` scope.

## Publishing

### Automatic Publishing

The library is automatically published to GitHub Packages when a release is created or when the workflow is manually triggered.

### Manual Publishing

To publish manually:

```bash
export USERNAME=your-github-username
export TOKEN=your-github-personal-access-token
./gradlew publishAllPublicationsToGitHubPackagesRepository
```

## Building

```bash
./gradlew build
```

## Testing

```bash
./gradlew test
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.