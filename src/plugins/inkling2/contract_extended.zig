//! Registrations for every remaining Inkling2 visit launch. The separate drift test uses the production launch
//! helpers, while gates 1-4 use the independent host formula and gate 5 checks every declared shape without a GPU.
//! The 70 normal variants retain their wide paths; 24 additional stream cases exercise the real 32-thread fallback.
//! Strict M1/M2 compiler qualification covers 46 fixed variants and those 24 fallbacks, not unmeasured wide caps.
const std = @import("std");
const m = @import("contract_math.zig");
const streams = @import("contract_streams.zig");
const prep = @import("contract_prep.zig");
const attention = @import("contract_attention.zig");
const experts = @import("contract_experts.zig");
const kernels = @import("kernels.zig");
const spec = m.spec;

const Mode = enum { normal, fallback, offline };

fn add(comptime mode: Mode, comptime drift: bool, set: *m.kc.Set, comptime T: type, comptime s: spec.Spec) !void {
    const adaptive = @hasDecl(T, "initFallback");
    if (mode == .fallback and !adaptive) return;
    const use_fallback = mode != .normal and adaptive;
    const stated = comptime blk: {
        var copy = s;
        if (use_fallback) copy.name = s.name ++ " fallback32";
        break :blk copy;
    };
    try set.add(T, if (use_fallback) T.initFallback(set.a) else T.init(set.a), stated, null);
    if (drift) {
        const k = set.kernels.items[set.kernels.items.len - 1];
        const p: *T = @ptrCast(@alignCast(k.ctx));
        for (s.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try set.a.dupe(u8, p.rig.outputBytes());
            defer set.a.free(ours);
            const actual = try p.wrapper();
            if (!std.mem.eql(u8, ours, actual)) std.debug.print("launch drift: {s} {d}x{d}x{d}\n", .{ s.name, shape.rows, shape.n, shape.k });
            try std.testing.expectEqualSlices(u8, ours, actual);
        }
    }
}

const nonlinear_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "normalization/softmax and stored intermediate roundings can hide a narrow sum; stress still checks every fp32 sum order and the device against the propagated float64 bound" };
const expert_sums: spec.Sums = .{};

fn two(sh: spec.Shape) usize {
    return 2 * m.elements(sh);
}

fn registrations(comptime mode: Mode, comptime drift: bool, set: *m.kc.Set) !void {
    inline for (.{ m.mlx.mlx_dtype.bfloat16, m.mlx.mlx_dtype.float32 }) |dt| {
        const suffix = if (dt == .bfloat16) " bf16" else " f32";
        inline for (.{ @as(usize, 0), 1, 3 }) |q| {
            try add(mode, drift, set, streams.Pre(dt, false, q), .{
                .name = "inkling2 pre " ++ std.fmt.comptimePrint("queries{d}", .{q}) ++ suffix,
                .accum = .f32,
                .depth = m.host.depthOne,
                .out = m.host.elemOf(dt),
                .outs = &.{ .{ .elem = m.host.elemOf(dt), .count = if (q > 1) m.three else m.elements }, .{ .elem = .f32, .count = m.elements } },
                .tiles = .{ .{ .size = 4 }, .{ .size = 2, .min = 0 }, .{ .size = 768, .step = 32, .min = 32 } },
                .served = &.{ .{ .rows = 8, .n = 2, .k = 2304 }, .{ .rows = 33, .n = 4, .k = 2304 }, .{ .rows = 1, .n = 0, .k = 2304 } },
                .sums = nonlinear_sums,
            });
        }
        try add(mode, drift, set, streams.Pre(dt, true, 3), .{
            .name = "inkling2 pre messages" ++ suffix,
            .accum = .f32,
            .depth = m.host.depthOne,
            .out = .f32,
            .outs = &.{ .{ .elem = .f32, .count = m.elements }, .{ .elem = .f32, .count = m.three }, .{ .elem = .f32, .count = m.four }, .{ .elem = m.host.elemOf(dt), .count = m.six } },
            .tiles = .{ .{ .size = 4 }, .{ .size = 2, .min = 0 }, .{ .size = 768, .step = 32, .min = 32 } },
            .served = &.{ .{ .rows = 8, .n = 2, .k = 2304 }, .{ .rows = 33, .n = 4, .k = 2304 } },
            .sums = nonlinear_sums,
        });
        inline for (.{ false, true }) |msg| inline for (.{ false, true }) |boundary| inline for (.{ false, true }) |seq| {
            try add(mode, drift, set, streams.Mid(dt, msg, boundary, seq), .{
                .name = "inkling2 mid " ++ (if (msg) "messages " else "plain ") ++ (if (boundary) "boundary " else "residual ") ++ (if (seq) "sequence" else "slots") ++ suffix,
                .accum = .f32,
                .depth = m.host.depthOne,
                .out = .f32,
                .outs = if (msg) &.{ .{ .elem = .f32, .count = m.elements }, .{ .elem = .f32, .count = m.elements }, .{ .elem = m.host.elemOf(dt), .count = two } } else &.{ .{ .elem = .f32, .count = m.elements }, .{ .elem = m.host.elemOf(dt), .count = m.elements }, .{ .elem = .f32, .count = m.elements } },
                .tiles = .{ .{ .size = 4 }, .{ .size = 2, .min = 0 }, .{ .size = 768, .step = 32, .min = 32 } },
                .served = &.{ .{ .rows = 8, .n = 2, .k = 2304 }, .{ .rows = 33, .n = 4, .k = 2304 }, .{ .rows = 1, .n = 0, .k = 2304 } },
                .sums = nonlinear_sums,
            });
        };
        inline for (.{ false, true }) |dynamic| inline for (.{ false, true }) |seq| {
            try add(mode, drift, set, prep.Prep(dt, dynamic, seq), .{
                .name = "inkling2 prep " ++ (if (dynamic) "dynamic " else "static ") ++ (if (seq) "sequence" else "slots") ++ suffix,
                .accum = .f32,
                .depth = m.host.depthOne,
                .out = m.host.elemOf(dt),
                .outs = &.{ .{ .elem = m.host.elemOf(dt), .count = prep.qCount }, .{ .elem = .f32, .count = prep.qCount }, .{ .elem = m.host.elemOf(dt), .count = m.elements } },
                .tiles = .{ .{}, .{ .size = 4 }, .{ .size = 128, .step = 32, .min = 32 } },
                .served = &.{ .{ .rows = 24, .n = 1, .k = 128 }, .{ .rows = 24, .n = 8, .k = 128 }, .{ .rows = 24, .n = 33, .k = 128 }, .{ .rows = 24, .n = 1, .k = 256 } },
                .sums = nonlinear_sums,
            });
        };
        inline for (.{ @as(usize, 0), 1024 }) |window| {
            try add(mode, drift, set, attention.Flash(dt, window), .{
                .name = "inkling2 relattn " ++ (if (window == 0) "global" else "sliding1024") ++ suffix,
                .accum = .f32,
                .depth = m.host.depthOne,
                .out = m.host.elemOf(dt),
                .tiles = .{ .{}, .{ .size = 32 }, .{ .size = 16, .min = 8 } },
                .served = &.{ .{ .rows = 8, .n = 33, .k = 1057 }, .{ .rows = 8, .n = 65, .k = 1090 }, .{ .rows = 2, .n = 33, .k = 4104 } },
                .sums = nonlinear_sums,
            });
        }
    }
    try add(mode, drift, set, attention.PromptGate, .{
        .name = "inkling2 gate (prompt projected f32)",
        .accum = .f32,
        .depth = m.host.depthOne,
        .sums = nonlinear_sums,
        .out = .f32,
        .served = &.{ .{ .rows = 3, .n = 33, .k = 2304 }, .{ .rows = 1, .n = 512, .k = 2304 } },
    });
    inline for (.{ .{ 0, 4, 16 }, .{ 1, 4, 32 }, .{ 2, 8, 32 }, .{ 3, 2, 64 }, .{ 3, 3, 64 }, .{ 3, 4, 64 }, .{ 3, 5, 64 }, .{ 3, 6, 64 }, .{ 3, 8, 64 }, .{ 3, 4, 32 }, .{ 3, 4, 128 } }) |fmt| {
        inline for (.{ experts.Kind.up, .down, .tail }) |kind| {
            const rows: usize = if (kind == .tail) 1 else 8;
            const n: usize = if (kind == .up) 1152 else 2304;
            const k: usize = if (kind == .up) 2304 else 1152;
            try add(mode, drift, set, experts.Packed(kind, fmt[0], fmt[1], fmt[2], rows, fmt[0] == 0), .{
                .name = "inkling2 " ++ (if (kind == .up) "ups " else "downs ") ++ std.fmt.comptimePrint("fmt{d} bits{d} g{d} ", fmt) ++ (if (kind == .tail) "tail" else "rows8"),
                .accum = .f32,
                .depth = m.host.depthOne,
                .ops = 32,
                .out = .bf16,
                .tiles = .{ .{}, .{ .size = 2, .step = 2, .min = 2 }, .{ .size = 512, .step = 128, .min = 128 } },
                .served = &.{ .{ .rows = 8, .n = n, .k = k }, .{ .rows = 8, .n = 64, .k = 512 } },
                .sums = expert_sums,
            });
        }
    }
}

pub fn register(set: *m.kc.Set) !void {
    try registrations(.normal, false, set);
}

test "kernel contract inkling2: added launches match the production wrappers byte for byte" {
    if (m.mlx.noGpuBackend() or m.mlx.virtualGpu() or !m.mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var set = m.kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try registrations(.normal, true, &set);
}

test "kernel contract inkling2: added launches fit every chip at every declared shape" {
    const a = std.testing.allocator;
    var set = m.kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var bad: usize = 0;
    for (set.kernels.items) |k| {
        const shapes = try spec.shapes(a, k.spec);
        defer a.free(shapes);
        const l = k.vt.launch.?(k.ctx);
        for (shapes) |sh| if (k.vt.geometry.?(k.ctx, sh)) |g| {
            bad += try m.kc.limits.metalAll(&.{l.record(k.spec.name, sh, g.threads())}, &out, a);
        };
    }
    if (bad > 0) std.debug.print("{s}", .{out.items});
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "kernel contract inkling2: adaptive stream fallback32 matches wrappers and passes gates 1-4" {
    if (m.mlx.noGpuBackend() or m.mlx.virtualGpu() or !m.mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const prior = kernels.exchangeStreamThreadsForTest(32);
    defer _ = kernels.exchangeStreamThreadsForTest(prior);
    var set = m.kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try registrations(.fallback, true, &set);
    try std.testing.expectEqual(@as(usize, 24), set.kernels.items.len);
    var limits_out: std.ArrayList(u8) = .empty;
    defer limits_out.deinit(set.a);
    var bad_limits: usize = 0;
    for (set.kernels.items) |k| {
        const shapes = try spec.shapes(set.a, k.spec);
        defer set.a.free(shapes);
        for (shapes) |sh| if (k.vt.geometry.?(k.ctx, sh)) |g| {
            try std.testing.expectEqual(@as(u32, 32), g.threads());
            const l = k.vt.launch.?(k.ctx);
            bad_limits += try m.kc.limits.metalAll(&.{l.record(k.spec.name, sh, g.threads())}, &limits_out, set.a);
        };
    }
    if (bad_limits != 0) std.debug.print("{s}", .{limits_out.items});
    try std.testing.expectEqual(@as(usize, 0), bad_limits);
    var tally: m.kc.gates.Tally = .{};
    try m.kc.gates.runAll(set.a, set.kernels.items, 0xc0de, &tally);
    try std.testing.expectEqual(@as(usize, 0), tally.problems());
    try std.testing.expectEqual(set.kernels.items.len, tally.passed);
}

// Wide streams are checked numerically above and probe every production instantiation's pipeline limit.
// Here the real launcher is forced to its terminal 32-thread fallback; all fixed launches retain their geometry.
// A missing compiler match is an error in this scoped audit, never a substitute for a verified cap.
test "kernel contract inkling2 offline: fixed launches and stream fallbacks have verified M1/M2 compiler caps" {
    const dir = @import("../../env.zig").get(.MLX_SERVE_KERNEL_CONTRACT_OFFLINE) orelse return error.SkipZigTest;
    if (!@import("builtin").os.tag.isDarwin() or m.mlx.noGpuBackend()) return error.SkipZigTest;
    var set = m.kc.Set.init(std.testing.allocator);
    defer set.deinit();
    const prior = kernels.exchangeStreamThreadsForTest(32);
    defer _ = kernels.exchangeStreamThreadsForTest(prior);
    try registrations(.offline, false, &set);
    try @import("../../sdk/testing/kernel_contract/offline.zig").auditStrict(std.testing.allocator, std.testing.io, &set, std.mem.span(dir));
}
