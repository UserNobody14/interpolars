set dotenv-load

default:
  @just --list


[group: 'tests']
[doc('Run the test suite')]
test:
    uv run pytest

[group: 'build']
[doc('Build the project')]
build:
    just clean
    # Debug build by default for stability/repro (avoids gcc/rustc ICEs seen at -O3).
    CARGO_BUILD_JOBS=1 CFLAGS=-O0 CXXFLAGS=-O0 uv run --group build maturin develop --uv

[group: 'build']
[doc('Build the project (release). This may stress compilers; use only if needed.')]
build-for-publish:
    just clean
    CARGO_BUILD_JOBS=1 CFLAGS=-O1 CXXFLAGS=-O1 uv run --group build maturin build --zig --release

[group: 'build']
[doc('Build the project (release). This may stress compilers; use only if needed.')]
build-release:
    just clean
    CARGO_BUILD_JOBS=1 CFLAGS=-O1 CXXFLAGS=-O1 uv run --group build maturin develop --uv --release

[group: 'build']
[doc('Build manylinux wheels')]
build-manylinux:
    just clean
    # Note: ghcr.io/pyo3/maturin keeps the Rust toolchain under /root/.cargo, so running as a
    # non-root uid/gid can't execute cargo (permission denied). Instead, build as root and then
    # chown build artifacts back to the host user to avoid root-owned outputs.
    docker run --rm \
      -v "$(pwd)":/io \
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
      --entrypoint sh \
      ghcr.io/pyo3/maturin \
      -lc 'set -eux; maturin build --release; chown -R "$HOST_UID:$HOST_GID" /io/target /io/dist /io/build 2>/dev/null || true'

[group: 'build']
[doc('Clean the project')]
clean:
    rm -rf target dist build
    cargo clean
    rm -rf **/*.so
    uv clean

[group: 'publish']
[doc('Publish the project')]
publish:
    export UV_PUBLISH_TOKEN=$PYPI_TOKEN && uv publish target/wheels/*

[group: 'tests']
[doc('Run the smoke test')]
smoke-test:
    uv run --isolated --no-project --with target/wheels/*.whl tests/smoke_test.py