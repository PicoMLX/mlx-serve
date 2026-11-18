//! `pico-lib`: libPicoMlxServe.dylib, a macOS library that carries its own private MLX and mlx-serve's MLX
//! text-generation engine over it (`src/embed_lib.zig`, `src/pico_engine.zig`). It links the static, hidden-visibility
//! mlx + mlx-c staged in lib/mlx-static by scripts/build-mlx-static.sh, and nothing of llama.cpp, ds4, the Neural
//! Engine, webp, stb_image or the Jinja renderer: the engine compiles as one unit with those engines off, and what
//! the unit still names of the rest is answered as unavailable (`src/pico_stubs.c`, `src/ane_stub.c`). The dylib and
//! mlx.metallib install side by side in zig-out/lib: MLX looks for the metallib in the directory of the image its own
//! code is in (`dladdr`), which is the dylib's.
//!
//! Apple's linker makes the final link. Zig 0.17's has no exported-symbols list, and its static libc++ leaves the
//! `operator new` / `operator delete` family as weak exports, which dyld would coalesce with the host's. Here MLX uses
//! the system libc++, and `-exported_symbols_list` leaves `exports` as the library's only symbols.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const libs = @import("libs.zig");

const stage = "lib/mlx-static/lib";

/// The library's whole C interface.
const exports = [_][]const u8{
    "pico_ms_abi_version", "pico_ms_build_info_v1", "pico_ms_mlx_version", "pico_ms_smoke",
    "pico_ms_memory",      "pico_ms_can_serve",     "pico_ms_load_bytes",  "pico_ms_open",
    "pico_ms_close",       "pico_ms_tokenize",      "pico_ms_detokenize",  "pico_ms_submit",
    "pico_ms_cancel",      "pico_ms_model_memory",  "pico_ms_trim",
};

/// mlx-c first: each archive resolves against the ones after it.
const archives = [_][]const u8{ "libmlxc.a", "libmlx.a", "libjaccl.a", "libgguflib.a" };

const frameworks = [_][]const u8{ "Metal", "Foundation", "QuartzCore", "Accelerate", "IOKit" };

/// C entry points of what the library leaves out, each answering "unavailable".
const stubs = [_][]const u8{ "src/pico_stubs.c", "src/ane_stub.c" };

const stream_csrc = "src/sdk/weights/streamed/csrc";

fn buildInfoModule(b: *std.Build) *std.Build.Module {
    const tool = b.addExecutable(.{ .name = "pico_build_info", .root_module = b.createModule(.{
        .root_source_file = b.path("build/pico_build_info.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    }) });
    const stamp = b.addRunArtifact(tool);
    stamp.setCwd(b.path("."));
    // A Git tag, index, or untracked source change can change the stamp without changing the Zig build graph.
    stamp.has_side_effects = true;
    const files = b.addWriteFiles();
    _ = files.addCopyFile(stamp.addOutputFileArg("build-info.json"), "build-info.json");
    return b.createModule(.{
        .root_source_file = files.add("pico_build_info.zig", "pub const json = @embedFile(\"build-info.json\");\n"),
    });
}

/// `src` compiled by Apple's clang, as the final link is, with hidden visibility: C++ here must share MLX's libc++.
fn compile(b: *std.Build, min: std.SemanticVersion, flags: []const []const u8, src: []const u8) std.Build.LazyPath {
    const cc = b.addSystemCommand(&.{ "xcrun", "--sdk", "macosx", "clang", "-c", "-arch", "arm64", "-O2", "-fvisibility=hidden", "-fvisibility-inlines-hidden" });
    cc.addArg(b.fmt("-mmacosx-version-min={d}.{d}", .{ min.major, min.minor }));
    cc.addArgs(&.{ "-I", stream_csrc, "-I", "lib/mlx-static/include", "-I", "lib/mlx-static/include/metal_cpp", "-I", "lib/mlxc-src" });
    cc.addArgs(flags);
    cc.addFileArg(b.path(src));
    cc.addArg("-o");
    return cc.addOutputFileArg(b.fmt("{s}.o", .{std.fs.path.stem(src)}));
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    if (builtin.os.tag != .macos or target.result.os.tag != .macos) return;

    const min = target.result.os.version_range.semver.min;
    // The engine's build options: one compile unit, MLX alone (no ds4, no llama.cpp, no Neural Engine), and nothing
    // that spawns a process (`mas`).
    const options = b.addOptions();
    options.addOption([]const u8, "version", config.readAppVersion(b) orelse "0.0.0-dev");
    options.addOption(bool, "mas", true);
    options.addOption(bool, "ios", false);
    options.addOption(bool, "units", false);
    options.addOption(bool, "ds4_engine", false);
    options.addOption(bool, "llama_engine", false);
    options.addOption(bool, "ane_engine", false);
    options.addOption([]const u8, "mlx_c_version", "unknown");
    options.addOption([]const u8, "ds4_commit", "unknown");
    options.addOption([]const u8, "llama_tag", "unknown");
    // Non-empty: the engine's build id then hashes this, not the bytes of the program that loaded the library.
    options.addOption([]const u8, "git_sha", "pico");
    options.addOption(bool, "slow_tests", false);

    const embedded: libs.Embedded = .init(b, target, optimize);
    const mod = b.createModule(.{
        .root_source_file = b.path("src/embed_lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "build_options", .module = options.createModule() },
            .{ .name = "pico_build_info", .module = buildInfoModule(b) },
            .{ .name = "mlx_stamp", .module = config.mlxStampModule(b, "lib/mlx-static/.version") },
            .{ .name = "mlx_steel_sources", .module = embedded.mlx_steel_sources },
            .{ .name = "opencode2_plugin", .module = embedded.opencode2_plugin },
            .{ .name = "agent_skills", .module = embedded.agent_skills },
            .{ .name = "jinja_c", .module = libs.addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize, "") },
            .{ .name = "stb", .module = libs.addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, "") },
            .{ .name = "webp", .module = libs.addCHeaderModule(b, b.path("src/webp_stub.h"), b.path("src"), target, optimize, "") },
        },
    });
    _ = libs.addExl3Module(b, mod, target, optimize);
    const obj = b.addObject(.{ .name = "pico_mlx_serve", .root_module = mod });
    obj.bundle_compiler_rt = true;

    var list: std.ArrayList(u8) = .empty;
    for (exports) |name| list.print(b.allocator, "_{s}\n", .{name}) catch @panic("OOM");
    const exports_file = b.addWriteFiles().add("pico.exports", list.items);

    const link = b.addSystemCommand(&.{ "xcrun", "--sdk", "macosx", "clang++", "-dynamiclib", "-arch", "arm64" });
    link.addArg(b.fmt("-mmacosx-version-min={d}.{d}", .{ min.major, min.minor }));
    link.addArgs(&.{ "-install_name", "@rpath/libPicoMlxServe.dylib" });
    // A weak export is a symbol dyld may coalesce with another image's: fail the link rather than ship one.
    link.addArg("-Wl,-no_weak_exports");
    link.addPrefixedFileArg("-Wl,-exported_symbols_list,", exports_file);
    // Only what the exported functions reach stays.
    link.addArg("-Wl,-dead_strip");
    link.addFileArg(obj.getEmittedBin());
    for (stubs) |src| link.addFileArg(compile(b, min, &.{ "-x", "c", "-std=c11" }, src));
    // The expert stream's C side (`libs.addStreamSources`), against the static stage's headers.
    link.addFileArg(compile(b, min, &.{ "-x", "c", "-std=c11", "-pthread" }, stream_csrc ++ "/q3_lookahead4_exl3.c"));
    link.addFileArg(compile(b, min, &.{ "-x", "objective-c++", "-fno-objc-arc" }, stream_csrc ++ "/q3_event_shim.mm"));
    for ([_][]const u8{ "/mlx_event_shim.cpp", "/mlx_alloc_shim.cpp" }) |src|
        link.addFileArg(compile(b, min, &.{ "-x", "c++", "-std=c++20", "-D_METAL_", "-DACCELERATE_NEW_LAPACK", "-Wno-deprecated-declarations" }, b.fmt("{s}{s}", .{ stream_csrc, src })));
    for (archives) |a| link.addFileArg(b.path(b.fmt("{s}/{s}", .{ stage, a })));
    for (frameworks) |f| link.addArgs(&.{ "-framework", f });
    link.addArg("-o");
    const dylib = link.addOutputFileArg("libPicoMlxServe.dylib");

    const step = b.step("pico-lib", "Build libPicoMlxServe.dylib: a C interface over a private, statically linked MLX");
    step.dependOn(&b.addInstallFileWithDir(dylib, .lib, "libPicoMlxServe.dylib").step);
    step.dependOn(&b.addInstallFileWithDir(b.path(stage ++ "/mlx.metallib"), .lib, "mlx.metallib").step);
}
