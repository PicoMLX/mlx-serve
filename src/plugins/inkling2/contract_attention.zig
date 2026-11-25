//! Prompt-only gate and relative flash attention, including partial query/key tiles and nonzero key starts.
const std = @import("std");
const m = @import("contract_math.zig");
const kernels = @import("kernels.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");
const ci = m.ci;
const Shape = m.Shape;
const spec = m.spec;
const metal = m.metal;
const mlx = m.mlx;
const prompt_launch = m.launch("gate", &.{ "A", "MS", "ST", "F", "G" }, &.{"Y"}, 0, false);

pub const PromptGate = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &prompt_launch,
    data: m.Data,
    shape: Shape = undefined,
    pub fn init(a: std.mem.Allocator) PromptGate {
        return .{ .rig = .init(a), .data = .init(a) };
    }
    pub fn deinit(s: *PromptGate) void {
        s.rig.deinit();
        s.data.deinit();
    }
    pub fn geometry(_: *const PromptGate, sh: Shape) ?metal.Geometry {
        if (sh.rows == 0 or sh.n <= kernels.small_rows or sh.k == 0 or sh.rows * sh.n * sh.k % 64 != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(sh.rows * sh.n * sh.k), 1, 1 }, .tg = .{ 64, 1, 1 } };
        g.out(&.{ ci(sh.rows), ci(sh.n), ci(sh.k) }, .float32);
        g.int("D", sh.k);
        g.int("RANK", 64);
        g.int("PROJ", 1);
        return g;
    }
    pub fn prepare(s: *PromptGate, sh: Shape, fill: spec.Fill, seed: u64) !bool {
        if (s.geometry(sh) == null) return false;
        s.shape = sh;
        s.data.reset(&s.rig);
        var p = std.Random.DefaultPrng.init(seed);
        const rng = p.random();
        const n = sh.rows * sh.n * sh.k;
        for ([_]usize{ 0, 1, 3, 4 }) |i| try s.data.draw(i, rng, n, .float32, -4, 4);
        try s.data.constant(2, sh.rows * sh.n * 64, 0);
        if (fill == .nonfinite) {
            m.host.plant(s.data.values[0], sh.rows * sh.n, sh.k);
            m.host.plant(s.data.values[3], sh.rows * sh.n, sh.k);
        }
        for ([_]usize{ 0, 1, 3, 4 }) |i| try s.data.bind(&s.rig, i, .float32, &.{ ci(sh.rows), ci(sh.n), ci(sh.k) });
        try s.data.bind(&s.rig, 2, .bfloat16, &.{ ci(sh.rows), ci(sh.n), 64 });
        return true;
    }
    pub fn run(s: *PromptGate) !void {
        try s.rig.launch(s.launch, s.geometry(s.shape).?, s.data.args[0..5]);
    }
    pub fn count(s: *const PromptGate) usize {
        return s.shape.rows * s.shape.n * s.shape.k;
    }
    fn outputs(s: *const PromptGate, want: []f64, mag: []f64, calc: m.Calc) !void {
        const x = s.data.values;
        for (0..want.len) |i| {
            const f = m.tanh(.exact(x[3][i]));
            const g = m.tanh(.exact(x[4][i]));
            const a = m.scale(m.add(.exact(1), .{ .v = -f.v, .e = f.e }), x[0][i]);
            const b = m.scale(g, x[1][i]);
            var v = calc.sum(&.{ a.v, b.v });
            v.e += a.e + b.e;
            m.emit(want, mag, i, v);
        }
    }
    pub fn reference(s: *const PromptGate, want: []f64, mag: []f64) !void {
        try s.outputs(want, mag, .{});
    }
    pub fn simulate(s: *const PromptGate, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        const mag = try s.rig.a.alloc(f64, out.len);
        defer s.rig.a.free(mag);
        try s.outputs(out, mag, .{ .sim = .{ .acc = acc, .order = order } });
    }
    pub fn wrapper(s: *PromptGate) ![]const u8 {
        var g = m.ops.Graph.init(s.rig.a, s.rig.stream());
        defer g.deinit();
        const x = s.data.args;
        return s.rig.bytesOf(&.{try kernels.gateMix(&g, x[0], x[1], x[2], x[3], x[4], true)});
    }
};

const D = 128;
const GQA = 3;
const R = 1024;
const BK: usize = 16;
const LOG2E: f64 = @as(f32, 1.4426950408889634);
const soft = attn.Soft{ .base2 = true, .p_unit = 0x1p-11, .p_abs = 0x1p-25 };
const flash_launch = blk: {
    var l = m.launch("relattn", &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" }, &.{"out"}, 2 * (D * (BK + 8) + BK * (D + 8)), false);
    l.contiguous = false;
    break :blk l;
};

pub fn Flash(comptime dt: mlx.mlx_dtype, comptime window: usize) type {
    return struct {
        const Self = @This();
        rig: metal.Rig,
        launch: *const metal.Launch = &flash_launch,
        data: m.Data,
        shape: Shape = undefined,
        pub fn init(a: std.mem.Allocator) Self {
            return .{ .rig = .init(a), .data = .init(a) };
        }
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            s.data.deinit();
        }
        pub fn geometry(_: *const Self, sh: Shape) ?metal.Geometry {
            if (sh.rows == 0 or sh.n == 0 or sh.k < sh.n or sh.k < 8) return null;
            var g = metal.Geometry{ .grid = .{ ci((sh.n + 31) / 32 * 128), ci(sh.rows * GQA), 1 }, .tg = .{ 128, 1, 1 } };
            g.out(&.{ 1, ci(sh.n), ci(sh.rows * GQA), D }, dt);
            g.dtype("T", dt);
            g.int("D", D);
            g.int("HQ", sh.rows * GQA);
            g.int("HKV", sh.rows);
            g.int("R", R);
            g.int("WINDOW", window);
            g.int("BQ", 32);
            g.int("TG", 128);
            return g;
        }
        pub fn prepare(s: *Self, sh: Shape, fill: spec.Fill, seed: u64) !bool {
            if (s.geometry(sh) == null) return false;
            s.shape = sh;
            s.data.reset(&s.rig);
            var p = std.Random.DefaultPrng.init(seed);
            const rng = p.random();
            const hq = sh.rows * GQA;
            try s.data.draw(0, rng, hq * sh.n * D, dt, -1, 1);
            try s.data.draw(1, rng, sh.rows * sh.k * D, dt, -1, 1);
            try s.data.draw(2, rng, sh.rows * sh.k * D, dt, -2, 2);
            try s.data.draw(3, rng, sh.n * hq * R, dt, -1, 1);
            try s.data.draw(4, rng, sh.n * hq, .float32, -3, 3);
            try s.data.constant(5, 1, 7);
            s.data.values[6] = try s.data.allocator().dupe(f64, &.{ @floatFromInt(sh.n), @floatFromInt(sh.k) });
            try s.data.constant(7, 1, @as(f32, 1 / @sqrt(@as(f32, D))));
            if (fill == .nonfinite) {
                m.host.plant(s.data.values[0], hq * sh.n, D);
                // Only the first KV head is poisoned; other heads prove the output is not unconditionally NaN.
                m.host.plant(s.data.values[2], 4, D);
            }
            try s.data.bind(&s.rig, 0, dt, &.{ 1, ci(hq), ci(sh.n), D });
            try s.data.bind(&s.rig, 1, dt, &.{ 1, ci(sh.rows), ci(sh.k), D });
            try s.data.bind(&s.rig, 2, dt, &.{ 1, ci(sh.rows), ci(sh.k), D });
            try s.data.bind(&s.rig, 3, dt, &.{ 1, ci(sh.n), ci(hq), R });
            try s.data.bind(&s.rig, 4, .float32, &.{ 1, ci(sh.n), ci(hq) });
            try s.data.bind(&s.rig, 5, .int32, &.{1});
            try s.data.bind(&s.rig, 6, .int32, &.{2});
            try s.data.bind(&s.rig, 7, .float32, &.{});
            return true;
        }
        pub fn run(s: *Self) !void {
            try s.rig.launch(s.launch, s.geometry(s.shape).?, s.data.args[0..8]);
        }
        pub fn count(s: *const Self) usize {
            return s.shape.n * s.shape.rows * GQA * D;
        }
        fn outputs(s: *const Self, want: []f64, mag: []f64, sim: attn.Sim) !void {
            var arena = std.heap.ArenaAllocator.init(s.rig.a);
            defer arena.deinit();
            const a = arena.allocator();
            const sh = s.shape;
            const hq = sh.rows * GQA;
            const x = s.data.values;
            const keys = try a.alloc(attn.Key, sh.k + BK);
            const ends = try a.alloc(usize, (sh.k + BK - 1) / BK);
            const q = try a.alloc(f64, D);
            const k = try a.alloc(f64, D);
            const terms = try a.alloc(f64, D);
            const eo = try a.alloc(f64, D);
            const vv = try a.alloc(f64, sh.k * D);
            const qs: f64 = @as(f32, @floatCast(x[7][0] * LOG2E));
            for (0..hq) |h| {
                const kh = h / GQA;
                for (vv, 0..) |*v, i| v.* = m.host.rt(.float16, x[2][kh * sh.k * D + i]);
                for (0..sh.n) |r| {
                    for (q, 0..) |*v, i| v.* = m.host.rt(.float16, x[0][(h * sh.n + r) * D + i]);
                    const q0 = r / 32 * 32;
                    const first_diag = sh.k - sh.n + q0;
                    const last_diag = sh.k - sh.n + @min(q0 + 32, sh.n) - 1;
                    const first = @max(@as(usize, 7), if (window > 0) first_diag + 1 -| window else 0) / BK * BK;
                    var n: usize = 0;
                    var nb: usize = 0;
                    var k0 = first;
                    while (k0 <= last_diag) : (k0 += BK) {
                        for (k0..@min(k0 + BK, sh.k)) |j| {
                            const diag = sh.k - sh.n + r;
                            const visible = j >= 7 and j <= diag and (window == 0 or diag - j < window);
                            if (!visible) {
                                keys[n] = .{ .s = -std.math.inf(f64), .on = false, .v = vv[j * D ..][0..D] };
                                n += 1;
                                continue;
                            }
                            for (k, 0..) |*v, i| v.* = m.host.rt(.float16, x[1][(kh * sh.k + j) * D + i]);
                            const dot = attn.dot(q, k, qs, D, 1, sim, terms);
                            const bias: f64 = if (diag - j < R) @as(f32, @floatCast(x[3][(r * hq + h) * R + diag - j] * LOG2E)) else 0;
                            const score = dot.v + bias;
                            keys[n] = .{ .s = score, .e = dot.e + m.host.U * (@abs(bias) + @abs(score)), .v = vv[j * D ..][0..D] };
                            n += 1;
                        }
                        ends[nb] = n;
                        nb += 1;
                    }
                    const off = (r * hq + h) * D;
                    const out = want[off..][0..D];
                    const row = attn.online(soft, .{ .m = @as(f32, @floatCast(x[4][r * hq + h] * LOG2E)), .l = 1 }, keys[0..n], ends[0..nb], out, eo);
                    attn.normalize(row, out, eo);
                    for (mag[off..][0..D], eo) |*v, e| v.* = m.host.allowance(e);
                }
            }
        }
        pub fn reference(s: *const Self, want: []f64, mag: []f64) !void {
            try s.outputs(want, mag, null);
        }
        pub fn simulate(s: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            const mag = try s.rig.a.alloc(f64, out.len);
            defer s.rig.a.free(mag);
            try s.outputs(out, mag, .{ .acc = acc, .order = order });
        }
        pub fn wrapper(s: *Self) ![]const u8 {
            var g = m.ops.Graph.init(s.rig.a, s.rig.stream());
            defer g.deinit();
            const x = s.data.args;
            return s.rig.bytesOf(&.{try kernels.flashRead(&g, x[0], x[1], x[2], x[3], x[4], x[5], ci(window))});
        }
    };
}
