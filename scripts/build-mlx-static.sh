#!/usr/bin/env bash
# Build the pinned mlx + mlx-c submodules as STATIC, hidden-visibility archives into lib/mlx-static/{lib,include},
# with mlx.metallib beside them. `zig build pico-lib` links these into libPicoMlxServe.dylib (build/pico.zig), a
# library that carries its own private MLX. Same pins, patches and 26.2 deployment target as scripts/build-mlx.sh;
# nothing under lib/mlx or lib/.mlx-build is touched.
set -euo pipefail

DEPLOYMENT_TARGET="${MLX_DEPLOYMENT_TARGET:-26.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLX_SRC="$REPO_ROOT/lib/mlx-src"
MLXC_SRC="$REPO_ROOT/lib/mlxc-src"
STAGE="$REPO_ROOT/lib/mlx-static"
BUILD_ROOT="$REPO_ROOT/lib/.mlx-static-build"
# MLX compiles MLX_METAL_PATH in as the last place it looks for mlx.metallib, and builds the metallib there. A
# scratch directory removed after the install leaves the copy beside the loading binary as the only one found.
METALLIB_SCRATCH="$BUILD_ROOT/metallib"

die() { echo "[build-mlx-static] ERROR: $*" >&2; exit 1; }

[ -f "$MLX_SRC/CMakeLists.txt" ] && [ -f "$MLXC_SRC/CMakeLists.txt" ] \
  || die "submodules missing — run: git submodule update --init lib/mlx-src lib/mlxc-src"
xcrun -sdk macosx metal --version >/dev/null 2>&1 \
  || die "Metal compiler unavailable: xcodebuild -downloadComponent MetalToolchain"

NCPU="$(sysctl -n hw.ncpu)"

# The patches scripts/build-mlx.sh applies, the same way (the files a patch touches are reset first), then the one
# only this build needs: without it a missing mlx.metallib traps the process instead of failing the call.
{
  cat "$SCRIPT_DIR/build-mlx-patches.txt"
  echo "lib/mlx-src patches/mlx-metallib-error-array-init.patch"
} | while read -r dir patch || [ -n "$dir" ]; do
  case "$dir" in ''|\#*) continue ;; esac
  src="$REPO_ROOT/$dir" patch="$REPO_ROOT/$patch"
  git -C "$src" checkout -- $(git -C "$src" apply --numstat -p1 "$patch" | cut -f3)
  git -C "$src" apply -p1 "$patch"
done

HIDDEN=(
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_C_VISIBILITY_PRESET=hidden
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
)

# ── mlx (C++ core + metallib) ────────────────────────────────────────────────
cmake -S "$MLX_SRC" -B "$BUILD_ROOT/mlx" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=OFF \
  "${HIDDEN[@]}" \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_METAL_PATH="$METALLIB_SCRATCH" \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
mkdir -p "$METALLIB_SCRATCH"
cmake --build "$BUILD_ROOT/mlx" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlx" >/dev/null
# gguflib is a helper archive MLX links privately and does not install; a static link needs it.
cp "$(find "$BUILD_ROOT/mlx" -name libgguflib.a | head -1)" "$STAGE/lib/"
rm -rf "$METALLIB_SCRATCH"

# ── mlx-c against the staged mlx ─────────────────────────────────────────────
# A static mlx's exported target names jaccl::jaccl, which MLXConfig.cmake does not load: load it first.
cmake -S "$MLXC_SRC" -B "$BUILD_ROOT/mlxc" \
  -DCMAKE_PROJECT_INCLUDE="$STAGE/lib/cmake/jaccl/jacclTargets.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=OFF \
  "${HIDDEN[@]}" \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$STAGE" \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlxc" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlxc" >/dev/null

for f in libmlx.a libmlxc.a libgguflib.a mlx.metallib; do
  [ -f "$STAGE/lib/$f" ] || die "$f missing from $STAGE/lib"
done
echo "mlx=$(git -C "$MLX_SRC" rev-parse --short=12 HEAD) mlxc=$(git -C "$MLXC_SRC" rev-parse --short=12 HEAD) target=$DEPLOYMENT_TARGET" > "$STAGE/.version"
echo "[build-mlx-static] staged lib/mlx-static OK: $(cat "$STAGE/.version")"
