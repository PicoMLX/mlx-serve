//! Q/K/V preparation: static/dynamic convs, sequential/batched state, one-row bias and prompt relative states.
const std = @import("std");
const m = @import("contract_math.zig");
const kernels = @import("kernels.zig");
const V = m.V;
const Shape = m.Shape;
const spec = m.spec;
const metal = m.metal;
const mlx = m.mlx;
const ci = m.ci;
const KT = 4;
const DR: usize = 32;
const EXTENT: usize = 128;
const KEYS: usize = EXTENT + 64;
const prep_launch = m.launch("prep", &.{ "QP", "GP", "RP", "KP", "VP", "KDY", "VDY", "ST", "KW", "VW", "QS", "RS", "RJ", "PR" }, &.{ "Q", "GATE", "K", "V", "BIAS", "REL" }, 256 * 4, false);

fn clip(calc: m.Calc, a: std.mem.Allocator, v: []const V, threshold: f64) !V {
    const terms = try a.alloc(f64, v.len);
    var e: f64 = 0;
    for (v, terms) |x, *t| {
        t.* = x.v * x.v;
        e += 2 * @abs(x.v) * x.e + x.e * x.e;
    }
    var n = calc.sum(terms);
    n.e += e;
    const rt = @sqrt(n.v);
    const root: V = .{ .v = rt, .e = n.e / (2 * @sqrt(@max(n.v - n.e, std.math.floatMin(f64)))) + 4 * m.host.U * @abs(rt) };
    const f = m.div(.exact(threshold), m.add(root, .exact(1e-6)));
    if (std.math.isNan(f.v)) return f;
    const center = @min(f.v, 1);
    return .{ .v = center, .e = @max(@abs(center - @min(f.v - f.e, 1)), @abs(@min(f.v + f.e, 1) - center)) };
}

pub fn qCount(sh: Shape) usize {
    return sh.rows * sh.n * sh.k;
}

pub fn Prep(comptime dt: mlx.mlx_dtype, comptime dynamic: bool, comptime sequence: bool) type {
    return struct {
        const Self = @This();
        rig: metal.Rig,
        launch: *const metal.Launch = &prep_launch,
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
            if (sh.rows == 0 or sh.n == 0 or sh.k == 0 or sh.k % 32 != 0 or sh.k > 256) return null;
            const wide = sh.n > 1;
            var g = metal.Geometry{ .grid = .{ ci(sh.rows * sh.k + 2 * sh.k + sh.rows * (if (wide) DR else KEYS)), ci(sh.n), 1 }, .tg = .{ ci(sh.k), 1, 1 } };
            g.out(if (wide) &.{ 1, ci(sh.n), ci(sh.rows), ci(sh.k) } else &.{ 1, ci(sh.rows), 1, ci(sh.k) }, dt);
            g.out(&.{ci(qCount(sh))}, .float32);
            g.out(&.{ 1, 1, ci(sh.n), ci(sh.k) }, dt);
            g.out(&.{ 1, 1, ci(sh.n), ci(sh.k) }, dt);
            g.out(if (wide) &.{1} else &.{ 1, ci(sh.rows), 1, KEYS }, if (wide) .float32 else dt);
            g.out(if (wide) &.{ 1, ci(sh.n), ci(sh.rows), DR } else &.{1}, if (wide) dt else .float32);
            g.unread = if (wide) 0b010000 else 0b100000;
            g.dtype("T", dt);
            g.dtype("WT", dt);
            g.dtype("S", dt);
            g.int("H", sh.rows);
            g.int("HD", sh.k);
            g.int("HKV", 1);
            g.int("DREL", DR);
            g.int("KT", KT);
            g.int("DYN", @intFromBool(dynamic));
            g.int("WIDE", @intFromBool(wide));
            g.int("SEQ", @intFromBool(sequence));
            return g;
        }
        pub fn prepare(s: *Self, sh: Shape, fill: spec.Fill, seed: u64) !bool {
            if (s.geometry(sh) == null) return false;
            s.shape = sh;
            s.data.reset(&s.rig);
            var p = std.Random.DefaultPrng.init(seed);
            const rng = p.random();
            try s.data.draw(0, rng, qCount(sh), dt, -2, 2);
            try s.data.draw(1, rng, qCount(sh), dt, -4, 4);
            try s.data.draw(2, rng, sh.n * sh.rows * DR, dt, -2, 2);
            for (3..5) |i| try s.data.draw(i, rng, sh.n * sh.k, dt, -2, 2);
            for (5..7) |i| try s.data.draw(i, rng, sh.n * KT, dt, -0.125, 0.125);
            try s.data.draw(7, rng, (if (sequence) @as(usize, 1) else sh.n) * (KT - 1) * 2 * sh.k, .float32, -1, 1);
            for (8..10) |i| try s.data.draw(i, rng, sh.k * KT, dt, -0.125, 0.125);
            try s.data.draw(10, rng, sh.rows * sh.k, dt, 0.5, 1.5);
            try s.data.draw(11, rng, sh.rows * DR, dt, 0.5, 1.5);
            try s.data.draw(12, rng, DR * EXTENT, .float32, -0.02, 0.02);
            s.data.values[13] = try s.data.allocator().dupe(f64, &.{ 1.5, 1.0, @as(f32, 1e-6), if (sh.n == 1) 1.25 else 1.0, @as(f32, 1.0 / @sqrt(@as(f32, if (dynamic) 2 * KT + 1 else KT + 1))), KEYS });
            if (fill == .nonfinite) {
                m.host.plant(s.data.values[0], sh.n * sh.rows, sh.k);
                m.host.plant(s.data.values[2], sh.n * sh.rows, DR);
                m.host.plant(s.data.values[3], sh.n, sh.k);
            }
            if (fill == .stress) {
                m.host.intsInto(rng, s.data.values[0], -32, 32);
                @memset(s.data.values[10], 1);
                @memset(s.data.values[11], 1);
                for (5..7) |i| @memset(s.data.values[i], 0);
            }
            try s.data.bind(&s.rig, 0, dt, &.{ 1, ci(sh.n), ci(sh.rows * sh.k) });
            try s.data.bind(&s.rig, 1, dt, &.{ 1, ci(sh.n), ci(sh.rows * sh.k) });
            try s.data.bind(&s.rig, 2, dt, &.{ 1, ci(sh.n), ci(sh.rows * DR) });
            for (3..5) |i| try s.data.bind(&s.rig, i, dt, &.{ 1, ci(sh.n), ci(sh.k) });
            for (5..7) |i| try s.data.bind(&s.rig, i, dt, &.{ 1, ci(sh.n), KT });
            try s.data.bind(&s.rig, 7, .float32, &.{ if (sequence) 1 else ci(sh.n), KT - 1, ci(2 * sh.k) });
            for (8..10) |i| try s.data.bind(&s.rig, i, dt, if (dynamic) &.{ KT, ci(sh.k) } else &.{ ci(sh.k), KT, 1 });
            try s.data.bind(&s.rig, 10, dt, &.{ ci(sh.rows), ci(sh.k) });
            try s.data.bind(&s.rig, 11, dt, &.{ ci(sh.rows), DR });
            try s.data.bind(&s.rig, 12, .float32, &.{ DR, EXTENT });
            try s.data.bind(&s.rig, 13, .float32, &.{6});
            return true;
        }
        pub fn run(s: *Self) !void {
            try s.rig.launch(s.launch, s.geometry(s.shape).?, &s.data.args);
        }
        pub fn count(s: *const Self) usize {
            const sh = s.shape;
            return 2 * qCount(sh) + 2 * sh.n * sh.k + sh.rows * (if (sh.n > 1) sh.n * DR else KEYS);
        }
        fn outputs(s: *const Self, want: []f64, mag: []f64, calc: m.Calc) !void {
            var arena = std.heap.ArenaAllocator.init(s.rig.a);
            defer arena.deinit();
            const a = arena.allocator();
            const sh = s.shape;
            const x = s.data.values;
            const d = sh.k;
            const nq = qCount(sh);
            const q = try a.alloc(V, d);
            const rel = try a.alloc(V, DR);
            const conv = try a.alloc(V, d);
            const terms = try a.alloc(f64, DR);
            for (0..sh.n) |r| {
                for (0..sh.rows) |h| {
                    for (q, 0..) |*v, i| v.* = m.mul(.exact(x[0][(r * sh.rows + h) * d + i]), .exact(x[10][h * d + i]));
                    const qf = try clip(calc, a, q, x[13][0]);
                    for (q, 0..) |v, i| {
                        const at = (r * sh.rows + h) * d + i;
                        var val = m.round(dt, m.mul(v, qf));
                        if (x[13][3] != 1) val = m.round(dt, m.scale(val, x[13][3]));
                        m.emit(want, mag, at, val);
                        m.emit(want, mag, nq + at, m.host.sigmoid(.exact(x[1][at])));
                    }
                    for (rel, 0..) |*v, i| v.* = m.mul(.exact(x[2][(r * sh.rows + h) * DR + i]), .exact(x[11][h * DR + i]));
                    const rf = try clip(calc, a, rel, x[13][1]);
                    for (rel) |*v| v.* = m.round(dt, m.mul(v.*, rf));
                    const off = 2 * nq + 2 * sh.n * d;
                    if (sh.n > 1) {
                        for (rel, 0..) |v, i| m.emit(want, mag, off + (r * sh.rows + h) * DR + i, v);
                    } else for (0..KEYS) |col| {
                        var val = V.exact(0);
                        if (col + EXTENT >= KEYS) {
                            var inherited: f64 = 0;
                            for (rel, terms, 0..) |v, *t, j| {
                                const w = x[12][j * EXTENT + col + EXTENT - KEYS];
                                t.* = v.v * w;
                                inherited += @abs(w) * v.e;
                            }
                            val = calc.sum(terms);
                            val.e += inherited;
                            val = m.round(dt, m.scale(val, x[13][3]));
                        }
                        m.emit(want, mag, off + h * KEYS + col, val);
                    }
                }
                for (0..2) |is_v| {
                    for (conv, 0..) |*v, i| {
                        var taps: [KT]f64 = undefined;
                        for (&taps, 0..) |*t, j| {
                            const old = if (sequence) r + j < KT - 1 else j + 1 < KT;
                            const at = if (sequence) r + j else r * (KT - 1) + j;
                            const val = if (old) x[7][at * 2 * d + is_v * d + i] else x[3 + is_v][(if (sequence) r + j - (KT - 1) else r) * d + i];
                            const w = if (dynamic) x[8 + is_v][(KT - 1 - j) * d + i] + x[5 + is_v][r * KT + KT - 1 - j] else x[8 + is_v][i * KT + j];
                            t.* = val * w;
                        }
                        var val = calc.sum(&taps);
                        if (!dynamic) val = m.round(dt, val);
                        v.* = m.round(dt, m.scale(m.add(val, .exact(x[3 + is_v][r * d + i])), x[13][4]));
                    }
                    const f = if (is_v == 1) try m.norm(calc, a, conv, x[13][2]) else try clip(calc, a, conv, x[13][0]);
                    for (conv, 0..) |v, i| m.emit(want, mag, 2 * nq + is_v * sh.n * d + r * d + i, m.round(dt, m.mul(v, f)));
                }
            }
        }
        pub fn reference(s: *const Self, want: []f64, mag: []f64) !void {
            try s.outputs(want, mag, .{});
        }
        pub fn simulate(s: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            const mag = try s.rig.a.alloc(f64, out.len);
            defer s.rig.a.free(mag);
            try s.outputs(out, mag, .{ .sim = .{ .acc = acc, .order = order } });
        }
        pub fn wrapper(s: *Self) ![]const u8 {
            var g = m.ops.Graph.init(s.rig.a, s.rig.stream());
            defer g.deinit();
            const x = s.data.args;
            var cfg: @import("config.zig").Config = undefined;
            cfg.num_attention_heads = @intCast(s.shape.rows);
            cfg.num_key_value_heads = 1;
            cfg.head_dim = @intCast(s.shape.k);
            cfg.d_rel = DR;
            cfg.sconv_kernel = KT;
            cfg.qk_clip = 1.5;
            cfg.rel_clip = 1;
            cfg.rms_norm_eps = 1e-6;
            var l: @import("layers.zig").Layer = undefined;
            l.k_conv = if (dynamic) .{ .dynamic = .{ .static = x[8], .proj = undefined } } else .{ .static = x[8] };
            l.v_conv = if (dynamic) .{ .dynamic = .{ .static = x[9], .proj = undefined } } else .{ .static = x[9] };
            l.q_scale = x[10];
            l.rel_scale = x[11];
            l.rel_proj_rev = x[12];
            const o = (try kernels.prep(&g, &l, &cfg, &.{ x[0], x[1], x[2], x[0], x[3], x[4] }, .{ x[5], x[6] }, x[7], @floatCast(s.data.values[13][3]), KEYS)).?;
            return s.rig.bytesOf(if (s.shape.n > 1) &.{ o.q, o.gate, o.k, o.v, o.rel } else &.{ o.q, o.gate, o.k, o.v, o.bias });
        }
    };
}
