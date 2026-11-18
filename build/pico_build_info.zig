//! Write a build stamp from the checkout the companion is compiled from.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.skip();
    const path = args.next() orelse std.process.fatal("usage: pico_build_info <out.json>", .{});
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = try buildInfoJSON(init) });
}

fn buildInfoJSON(init: std.process.Init) ![]const u8 {
    const identity = gitOutput(init, &.{ "git", "rev-parse", "--show-prefix", "--verify", "HEAD^{commit}" }) orelse return "";
    // A nonempty prefix means Git found a parent checkout rather than these sources.
    if (identity.len == 0 or identity[0] != '\n') return "";
    const revision = std.mem.trim(u8, identity[1..], " \t\r\n");
    const tag = if (gitOutput(init, &.{ "git", "describe", "--tags", "--exact-match", revision })) |stdout| blk: {
        const name = std.mem.trim(u8, stdout, "\r\n");
        break :blk if (std.unicode.utf8ValidateSlice(name)) name else null;
    } else null;
    const status = gitOutput(init, &.{ "git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none" });
    const dirty = if (status) |stdout| stdout.len != 0 else null;
    return std.json.Stringify.valueAlloc(init.arena.allocator(), .{ .revision = revision, .tag = tag, .dirty = dirty }, .{
        .emit_null_optional_fields = false,
    });
}

fn gitOutput(init: std.process.Init, argv: []const []const u8) ?[]const u8 {
    const result = std.process.run(init.arena.allocator(), init.io, .{ .argv = argv }) catch return null;
    return switch (result.term) {
        .exited => |code| if (code == 0) result.stdout else null,
        else => null,
    };
}
