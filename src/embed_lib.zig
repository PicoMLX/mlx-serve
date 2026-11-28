//! Root of libPicoMlxServe.dylib (`zig build pico-lib`, build/pico.zig): a C interface over an MLX that lives inside
//! the library, and over mlx-serve's MLX text-generation engine on top of it (`pico_engine.zig`). mlx + mlx-c are
//! linked in statically and the final link exports only the `pico_ms_*` functions, so a host process that already
//! holds another MLX (Swift MLX, linked statically) keeps its own: the two copies share no symbol, no allocator and
//! no Metal library.
//!
//! Loading the library runs none of this code and changes nothing process-wide: no signal handler, no memory or
//! wired limit, no environment variable. mlx-c's error handler (state inside this library's mlx-c) is set on the first
//! call, since its default prints and calls `exit`.

const std = @import("std");
const os_mutex = @import("sdk/util/os_mutex.zig");

// What Sushi's EXL3 module reads from its host root (`mlx_host`), as `main.zig` declares them.
pub const env = @import("env.zig");
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const log = @import("sdk/util/log.zig");

/// Marks a library inside another program: the engine reads none of the user's mlx-serve settings (`~/.mlx-serve`).
pub const mlx_serve_embedded = true;

test {
    _ = @import("pico_engine.zig");
}

/// The test process caps MLX's buffer pool as the library does on open; `build/test_runner.zig` calls it first.
fn testProcessInit() callconv(.c) void {
    @import("mlx/state.zig").applyMlxCacheLimit();
}

comptime {
    if (@import("builtin").is_test) @export(&testProcessInit, .{ .name = "msv_test_process_init" });
    _ = @import("pico_engine.zig");
    // One compile unit: the MLX engine's table (`mlx_serve_mlx_api`, which its own code calls through) is here.
    _ = @import("engines/mlx/api.zig");
}

// Prints the message and traps: a panic in a library has no business symbolizing the host's stack.
pub const panic = std.debug.simple_panic;

var init_mutex: os_mutex.Mutex = .{};
var initialized: std.atomic.Value(bool) = .init(false);
/// The MLX version, copied out of mlx-c once; empty when mlx-c could not report it.
var version_buf: [64:0]u8 = @splat(0);

pub fn ensureInit() void {
    if (initialized.load(.acquire)) return;
    init_mutex.lock();
    defer init_mutex.unlock();
    if (initialized.load(.monotonic)) return;
    // The engine's handler: it keeps the first failure for the call that reads it and logs each one.
    mlx.installErrorHandler();
    // The engine's warnings and errors reach stderr; its running account of loads and requests does not.
    log.setLevel(.warn);
    var str = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(str);
    if (mlx.mlx_version(&str) == 0) {
        const text = std.mem.span(mlx.mlx_string_data(str));
        const len = @min(text.len, version_buf.len);
        @memcpy(version_buf[0..len], text[0..len]);
    }
    initialized.store(true, .release);
}

pub export fn pico_ms_abi_version() i32 {
    return 2;
}

/// Source build information as static UTF-8 JSON, or null for an unstamped build.
export fn pico_ms_build_info_v1() ?[*:0]const u8 {
    const json = @import("pico_build_info").json;
    return if (json.len == 0) null else json;
}

/// The version of the MLX inside this library; static storage.
export fn pico_ms_mlx_version() [*:0]const u8 {
    ensureInit();
    return &version_buf;
}

/// Builds A, an n×n float32 matrix of 1/n, on the default GPU stream, multiplies P = A by A `rounds` times (A is
/// idempotent, so P stays A), evaluates it and returns the sum of P's elements: n. NaN on any failure.
export fn pico_ms_smoke(n: i32, rounds: i32) f64 {
    return smoke(n, rounds) catch {
        // Its own failure: nothing later should read it.
        var buf: [64]u8 = undefined;
        _ = mlx.takeError(&buf);
        return std.math.nan(f64);
    };
}

fn smoke(n: i32, rounds: i32) !f64 {
    if (n <= 0 or rounds < 0) return error.InvalidArgument;
    ensureInit();
    // The calling thread's own default GPU stream: MLX keeps default streams per thread.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Empty when MLX could not bring the GPU up (no Metal device, or no mlx.metallib beside the library).
    if (s.ctx == null) return error.MlxError;

    const fill = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(n)));
    defer _ = mlx.mlx_array_free(fill);
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    const shape = [_]c_int{ n, n };
    try mlx.check(mlx.mlx_full(&a, &shape, shape.len, fill, .float32, s));

    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_array_set(&p, a));
    for (0..@intCast(rounds)) |_| try mlx.check(mlx.mlx_matmul(&p, p, a, s));

    var total = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(total);
    try mlx.check(mlx.mlx_sum(&total, p, false, s));
    try mlx.check(mlx.mlx_array_eval(total));
    var value: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&value, total));
    return value;
}

/// This MLX's active, cache and peak bytes into `out[0..3]`. 0 on success.
pub export fn pico_ms_memory(out: ?*[3]i64) i32 {
    const o = out orelse return -1;
    ensureInit();
    var bytes: [3]usize = undefined;
    if (mlx.mlx_get_active_memory(&bytes[0]) != 0) return -1;
    if (mlx.mlx_get_cache_memory(&bytes[1]) != 0) return -1;
    if (mlx.mlx_get_peak_memory(&bytes[2]) != 0) return -1;
    for (o, bytes) |*dst, b| dst.* = @intCast(b);
    return 0;
}
