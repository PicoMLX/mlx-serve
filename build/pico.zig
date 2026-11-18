//! `pico-lib`: libPicoMlxServe.dylib, a macOS library that carries its own private MLX (`src/embed_lib.zig`). It links
//! the static, hidden-visibility mlx + mlx-c staged in lib/mlx-static by scripts/build-mlx-static.sh, and nothing of
//! llama.cpp, ds4, ANE or webp. The dylib and mlx.metallib install side by side in zig-out/lib: MLX looks for the
//! metallib in the directory of the image its own code is in (`dladdr`), which is the dylib's.
//!
//! Apple's linker makes the final link. Zig 0.17's has no exported-symbols list, and its static libc++ leaves the
//! `operator new` / `operator delete` family as weak exports, which dyld would coalesce with the host's. Here MLX uses
//! the system libc++, and `-exported_symbols_list` leaves `exports` as the library's only symbols.

const std = @import("std");
const builtin = @import("builtin");

const stage = "lib/mlx-static/lib";

/// The library's whole C interface.
const exports = [_][]const u8{ "pico_ms_abi_version", "pico_ms_mlx_version", "pico_ms_smoke", "pico_ms_memory" };

/// mlx-c first: each archive resolves against the ones after it.
const archives = [_][]const u8{ "libmlxc.a", "libmlx.a", "libjaccl.a", "libgguflib.a" };

const frameworks = [_][]const u8{ "Metal", "Foundation", "QuartzCore", "Accelerate", "IOKit" };

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    if (builtin.os.tag != .macos or target.result.os.tag != .macos) return;

    const obj = b.addObject(.{ .name = "pico_mlx_serve", .root_module = b.createModule(.{
        .root_source_file = b.path("src/embed_lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    obj.bundle_compiler_rt = true;

    var list: std.ArrayList(u8) = .empty;
    for (exports) |name| list.print(b.allocator, "_{s}\n", .{name}) catch @panic("OOM");
    const exports_file = b.addWriteFiles().add("pico.exports", list.items);

    const min = target.result.os.version_range.semver.min;
    const link = b.addSystemCommand(&.{ "xcrun", "--sdk", "macosx", "clang++", "-dynamiclib", "-arch", "arm64" });
    link.addArg(b.fmt("-mmacosx-version-min={d}.{d}", .{ min.major, min.minor }));
    link.addArgs(&.{ "-install_name", "@rpath/libPicoMlxServe.dylib" });
    // A weak export is a symbol dyld may coalesce with another image's: fail the link rather than ship one.
    link.addArg("-Wl,-no_weak_exports");
    link.addPrefixedFileArg("-Wl,-exported_symbols_list,", exports_file);
    link.addFileArg(obj.getEmittedBin());
    for (archives) |a| link.addFileArg(b.path(b.fmt("{s}/{s}", .{ stage, a })));
    for (frameworks) |f| link.addArgs(&.{ "-framework", f });
    link.addArg("-o");
    const dylib = link.addOutputFileArg("libPicoMlxServe.dylib");

    const step = b.step("pico-lib", "Build libPicoMlxServe.dylib: a C interface over a private, statically linked MLX");
    step.dependOn(&b.addInstallFileWithDir(dylib, .lib, "libPicoMlxServe.dylib").step);
    step.dependOn(&b.addInstallFileWithDir(b.path(stage ++ "/mlx.metallib"), .lib, "mlx.metallib").step);
}
