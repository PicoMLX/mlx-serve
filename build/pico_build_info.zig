//! Build-time source identity for the Pico library, independent of the executable that loads it.
const std = @import("std");

const Info = struct { revision: []const u8, tag: ?[]const u8, dirty: ?bool };
pub const Captured = struct { options: *std.Build.Step.Options, git_sha: []const u8 };

pub fn capture(b: *std.Build) Captured {
    b.graph.poisonCache(); // Git's revision and worktree state are outside Zig's tracked file graph.
    const info = identity(
        git(b, &.{ "rev-parse", "--show-prefix" }),
        git(b, &.{ "rev-parse", "--verify", "HEAD" }),
        git(b, &.{ "describe", "--tags", "--exact-match", "HEAD" }),
        git(b, &.{ "status", "--porcelain", "--untracked-files=all" }),
    );
    const options = b.addOptions();
    const json = if (info) |value| std.json.Stringify.valueAlloc(b.allocator, value, .{}) catch @panic("OOM") else null;
    options.addOption(?[]const u8, "json", json);
    return .{
        .options = options,
        .git_sha = if (info) |value| b.fmt("{s}{s}", .{ value.revision, dirtySuffix(value.dirty) }) else "pico-source-unknown",
    };
}

fn git(b: *std.Build, args: []const []const u8) ?[]const u8 {
    const argv = std.mem.concat(b.allocator, []const u8, &.{
        &.{ "git", "-C", b.root.root_dir.path orelse "." }, args,
    }) catch @panic("OOM");
    var code: u8 = 0;
    const output = b.runAllowFail(argv, &code, .ignore) catch return null;
    return if (code == 0) output else null;
}

fn trim(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, "\r\n");
}

fn identity(prefix_output: ?[]const u8, revision_output: ?[]const u8, tag_output: ?[]const u8, status_output: ?[]const u8) ?Info {
    // A source archive nested in an unrelated checkout must not borrow that checkout's identity.
    if (trim(prefix_output orelse return null).len != 0) return null;
    const revision = trim(revision_output orelse return null);
    if (revision.len != 40 and revision.len != 64) return null;
    for (revision) |byte| if (!std.ascii.isHex(byte)) return null;
    const tag = if (tag_output) |text| trim(text) else "";
    return .{
        .revision = revision,
        .tag = if (tag.len == 0) null else tag,
        .dirty = if (status_output) |text| trim(text).len != 0 else null,
    };
}

fn dirtySuffix(dirty: ?bool) []const u8 {
    return if (dirty) |value| (if (value) "-dirty" else "") else "-dirty-unknown";
}

pub fn tests(b: *std.Build) void {
    const module = b.createModule(.{ .root_source_file = b.path("build/pico_build_info.zig"), .target = b.graph.host });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    b.step("pico-build-info-test", "Pico source identity and archive metadata, without MLX or a GPU").dependOn(&run.step);
}

test "Pico metadata requires the library's own checkout and a full Git revision" {
    const revision: [40]u8 = @splat('a');
    try std.testing.expect(identity(null, null, null, null) == null);
    try std.testing.expect(identity("archive/", &revision, null, "") == null);
    try std.testing.expect(identity("", "2201ec9a\n", null, null) == null);
    const invalid: [40]u8 = @splat('x');
    try std.testing.expect(identity("", &invalid, null, null) == null);
    const revision256: [64]u8 = @splat('a');
    const sha256 = identity("", &revision256, null, null).?;
    try std.testing.expectEqualStrings(&revision256, sha256.revision);
    try std.testing.expect(sha256.tag == null);
    try std.testing.expect(sha256.dirty == null);
}

test "Pico metadata distinguishes exact tags, changed sources and unknown status" {
    const revision: [40]u8 = @splat('a');
    const clean = identity("", &revision, "v2026.11.23\n", "").?;
    try std.testing.expectEqualStrings(&revision, clean.revision);
    try std.testing.expectEqualStrings("v2026.11.23", clean.tag.?);
    try std.testing.expectEqual(false, clean.dirty.?);
    const changed = identity("", &revision, null, " M src/embed_lib.zig\n?? new-file\n").?;
    try std.testing.expect(changed.tag == null);
    try std.testing.expectEqual(true, changed.dirty.?);
    const unknown = identity("", &revision, "", null).?;
    try std.testing.expect(unknown.tag == null);
    try std.testing.expect(unknown.dirty == null);
    try std.testing.expectEqualStrings("", dirtySuffix(clean.dirty));
    try std.testing.expectEqualStrings("-dirty", dirtySuffix(changed.dirty));
    try std.testing.expectEqualStrings("-dirty-unknown", dirtySuffix(unknown.dirty));
}

test "Pico metadata JSON preserves an exact tag requiring escaping" {
    const revision: [40]u8 = @splat('b');
    const info = identity("", &revision, "release-\"quoted\"", "").?;
    const json = try std.json.Stringify.valueAlloc(std.testing.allocator, info, .{});
    defer std.testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(Info, std.testing.allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(info.revision, parsed.value.revision);
    try std.testing.expectEqualStrings(info.tag.?, parsed.value.tag.?);
    try std.testing.expectEqual(false, parsed.value.dirty.?);
}
