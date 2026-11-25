//! The stream/depth-attention launches, including the later-pass message arms. All reductions below are on
//! float64 host inputs; the simulator changes only their accumulator, never the reference or the GPU result.
const std = @import("std");
const m = @import("contract_math.zig");
const kernels = @import("kernels.zig");
const config = @import("config.zig");
const V = m.V;
const Shape = m.Shape;
const spec = m.spec;
const metal = m.metal;
const mlx = m.mlx;
const ci = m.ci;

const pre_launch = m.launch("pre", &.{ "HIN", "BLK", "QY", "NW0", "NW1", "NW2", "BANK", "BPART", "MQ", "PR" }, &.{ "X", "NORMED", "A", "MSGO", "BOTH" }, 4 * 24 * (1 + 3 * 25 + 4 * 25 + 7), true);

pub fn Pre(comptime dt: mlx.mlx_dtype, comptime msg: bool, comptime queries: usize) type {
    return struct {
        const Self = @This();
        const J: usize = if (queries > 1) 3 else 1;
        const QC: usize = @max(queries, 1);
        pub const fallback_launch = blk: {
            var l = pre_launch;
            l.adapts = false;
            l.offline = true;
            l.m1 = 32;
            l.m2 = 32;
            break :blk l;
        };
        rig: metal.Rig,
        launch: *const metal.Launch = &pre_launch,
        data: m.Data,
        shape: Shape = undefined,
        adaptive_probed: bool = false,
        last_run_threads: u32 = 0,
        pub fn init(a: std.mem.Allocator) Self {
            return .{ .rig = .init(a), .data = .init(a) };
        }
        pub fn initFallback(a: std.mem.Allocator) Self {
            return .{ .rig = .init(a), .data = .init(a), .launch = &fallback_launch };
        }
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            s.data.deinit();
        }
        pub fn geometry(_: *const Self, sh: Shape) ?metal.Geometry {
            if (sh.rows == 0 or sh.n > 24 or sh.k == 0) return null;
            const tg = kernels.streamGroup(@intCast(sh.k)) orelse return null;
            var g = metal.Geometry{ .grid = .{ ci(sh.k), ci(sh.rows), 1 }, .tg = .{ tg, 1, 1 } };
            g.out(if (msg) &.{1} else &.{ J, ci(sh.rows), ci(sh.k) }, if (msg) .float32 else dt);
            g.out(&.{ 1, ci(sh.rows), ci(sh.k) }, .float32);
            g.out(if (msg) &.{ 3, ci(sh.rows), ci(sh.k) } else &.{1}, .float32);
            g.out(if (msg) &.{ 4, ci(sh.rows), ci(sh.k) } else &.{1}, .float32);
            g.out(if (msg) &.{ 3, ci(sh.rows), ci(2 * sh.k) } else &.{1}, if (msg) dt else .float32);
            g.unread = if (msg) 0b00001 else 0b11100;
            g.dtype("T", dt);
            g.dtype("GT", dt);
            g.int("TG", tg);
            g.int("D", sh.k);
            g.int("Q", queries);
            g.int("J", J);
            g.int("NB", sh.n);
            g.int("MSG", @intFromBool(msg));
            g.int("NBK", if (msg) sh.n else 0);
            return g;
        }
        pub fn prepare(s: *Self, sh: Shape, fill: spec.Fill, seed: u64) !bool {
            if (s.geometry(sh) == null) return false;
            s.shape = sh;
            s.adaptive_probed = false;
            s.last_run_threads = 0;
            s.data.reset(&s.rig);
            var p = std.Random.DefaultPrng.init(seed);
            const rng = p.random();
            const d = sh.k;
            const rows = sh.rows;
            try s.data.draw(0, rng, rows * d, dt, -2, 2);
            try s.data.draw(1, rng, @max(rows * sh.n * d, 1), .float32, -1, 1);
            try s.data.draw(2, rng, QC * d, .float32, -0.02, 0.02);
            for (3..6) |i| try s.data.draw(i, rng, d, dt, 0.5, 1.5);
            try s.data.draw(6, rng, @max(rows * sh.n * d, 1), .float32, -1, 1);
            try s.data.draw(7, rng, rows * d, .float32, -1, 1);
            try s.data.draw(8, rng, 4 * d, .float32, -0.02, 0.02);
            try s.data.constant(9, 1, @as(f32, 1e-6));
            if (fill == .nonfinite) m.host.plant(s.data.values[0], rows, d);
            if (fill == .stress) {
                m.host.intsInto(rng, s.data.values[0], -32, 32);
                // Zero queries give equal source weights. Integer inputs prove the RMS reductions without
                // introducing an inexact dot or a saturated softmax before those sums.
                @memset(s.data.values[2], 0);
                @memset(s.data.values[8], 0);
                for (3..6) |i| @memset(s.data.values[i], 1);
            }
            try s.data.bind(&s.rig, 0, dt, &.{ 1, ci(rows), ci(d) });
            try s.data.bind(&s.rig, 1, .float32, if (sh.n == 0) &.{1} else &.{ 1, ci(rows), ci(sh.n), ci(d) });
            try s.data.bind(&s.rig, 2, .float32, &.{ ci(d), ci(QC) });
            for (3..6) |i| try s.data.bind(&s.rig, i, dt, &.{ci(d)});
            try s.data.bind(&s.rig, 6, .float32, if (sh.n == 0) &.{1} else &.{ 1, ci(rows), ci(sh.n), ci(d) });
            try s.data.bind(&s.rig, 7, .float32, &.{ 1, ci(rows), ci(d) });
            try s.data.bind(&s.rig, 8, .float32, &.{ ci(d), 4 });
            try s.data.bind(&s.rig, 9, .float32, &.{1});
            return true;
        }
        pub fn run(s: *Self) !void {
            // The production wrapper probes each instantiation and may narrow the shared stream group.
            // Recompute geometry only after that probe, including when a numerical gate calls run directly.
            if (s.launch.adapts and !s.adaptive_probed) _ = try s.wrapper();
            const g = s.geometry(s.shape).?;
            s.last_run_threads = g.threads();
            try s.rig.launch(s.launch, g, s.data.args[0..10]);
        }
        pub fn count(s: *const Self) usize {
            return (if (msg) 14 else J + 1) * s.shape.rows * s.shape.k;
        }
        fn outputs(s: *const Self, want: []f64, mag: []f64, calc: m.Calc) !void {
            var arena = std.heap.ArenaAllocator.init(s.rig.a);
            defer arena.deinit();
            const a = arena.allocator();
            const d = s.shape.k;
            const rows = s.shape.rows;
            const nb = s.shape.n;
            const sd = rows * d;
            const x = s.data.values;
            const terms = try a.alloc(f64, d);
            const live = try a.alloc(V, d);
            const normed = try a.alloc(V, d);
            const sources = try a.alloc(V, (QC + 4) * d);
            const logits = try a.alloc(V, nb + 1);
            for (0..rows) |r| {
                for (live, 0..) |*v, i| v.* = .exact(x[0][r * d + i]);
                const hn = try m.norm(calc, a, live, 1e-6);
                for (normed, live, 0..) |*v, h, i| {
                    v.* = m.mul(h, hn);
                    m.emit(want, mag, (if (msg) 0 else J * sd) + r * d + i, v.*);
                }
                for (0..QC) |q| {
                    const dest = sources[q * d ..][0..d];
                    if (queries == 0) {
                        @memcpy(dest, live);
                        continue;
                    }
                    for (0..nb + 1) |b| {
                        var inherited: f64 = 0;
                        for (terms, 0..) |*t, i| {
                            const v: V = if (b == nb) normed[i] else .exact(x[1][(r * nb + b) * d + i]);
                            const weight = x[2][i * queries + q];
                            t.* = v.v * weight;
                            inherited += @abs(weight) * v.e;
                        }
                        logits[b] = calc.sum(terms);
                        logits[b].e += inherited;
                    }
                    const weights = try m.softmax(calc, a, logits);
                    for (dest, 0..) |*v, i| {
                        var mixed: [25]V = undefined;
                        for (weights, 0..) |w, b| mixed[b] = m.mul(w, if (b == nb) normed[i] else .exact(x[1][(r * nb + b) * d + i]));
                        v.* = calc.sumValues(mixed[0..weights.len], terms[0..weights.len]);
                    }
                }
                if (msg) for (0..4) |q| {
                    for (0..nb + 1) |b| {
                        for (terms, 0..) |*t, i| t.* = (if (b == nb) x[7][r * d + i] else x[6][(r * nb + b) * d + i]) * x[8][i * 4 + q];
                        logits[b] = calc.sum(terms);
                    }
                    const weights = try m.softmax(calc, a, logits);
                    const dest = sources[(QC + q) * d ..][0..d];
                    for (dest, 0..) |*v, i| {
                        var mixed: [25]V = undefined;
                        for (weights, 0..) |w, b| mixed[b] = m.scale(w, if (b == nb) x[7][r * d + i] else x[6][(r * nb + b) * d + i]);
                        v.* = calc.sumValues(mixed[0..weights.len], terms[0..weights.len]);
                        m.emit(want, mag, 4 * sd + (q * rows + r) * d + i, v.*);
                    }
                };
                for (0..if (msg) 3 else J) |j| {
                    const v = sources[@min(j, QC - 1) * d ..][0..d];
                    const rn = try m.norm(calc, a, v, if (msg) 1e-6 else x[9][0]);
                    const vm = if (msg) sources[(QC + j) * d ..][0..d] else v;
                    const rm = if (msg) try m.norm(calc, a, vm, 1e-6) else rn;
                    for (v, 0..) |h, i| {
                        if (msg) {
                            m.emit(want, mag, sd + (j * rows + r) * d + i, h);
                            m.emit(want, mag, 8 * sd + (j * rows + r) * 2 * d + i, m.round(dt, m.mul(h, rn)));
                            m.emit(want, mag, 8 * sd + (j * rows + r) * 2 * d + d + i, m.round(dt, m.mul(vm[i], rm)));
                        } else m.emit(want, mag, (j * rows + r) * d + i, m.scale(m.mul(h, rn), x[3 + j][i]));
                    }
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
            var cfg: config.Config = undefined;
            cfg.hidden_size = @intCast(s.shape.k);
            cfg.rms_norm_eps = 1e-6;
            const x = s.data.args;
            const o = try kernels.streams(&g, &cfg, x[0], if (s.shape.n > 0) x[1] else null, if (queries > 0) x[2] else .{ .ctx = null }, .{ x[3], x[4], x[5] }, J, if (msg) .{ .bank = x[6], .partial = x[7], .query = x[8], .gt = dt } else null);
            const bytes = try s.rig.bytesOf(if (msg) &.{ o[1], o[2], o[3], o[4] } else &.{ o[0], o[1] });
            s.adaptive_probed = true;
            return bytes;
        }
    };
}

const KT = 4;
const KV = 64;
const mid_launch = m.launch("mid", &.{ "OP", "ST", "AW", "HIN", "NORMED", "BLK", "QY", "PN", "MSG3", "PR" }, &.{ "RES", "X", "X32", "M", "BOTH" }, 4 * 24 * (1 + 26 + 1 + 1), true);

pub fn Mid(comptime dt: mlx.mlx_dtype, comptime msg: bool, comptime boundary: bool, comptime sequence: bool) type {
    return struct {
        const Self = @This();
        pub const fallback_launch = blk: {
            var l = mid_launch;
            l.adapts = false;
            l.offline = true;
            l.m1 = 32;
            l.m2 = 32;
            break :blk l;
        };
        rig: metal.Rig,
        launch: *const metal.Launch = &mid_launch,
        data: m.Data,
        shape: Shape = undefined,
        adaptive_probed: bool = false,
        last_run_threads: u32 = 0,
        pub fn init(a: std.mem.Allocator) Self {
            return .{ .rig = .init(a), .data = .init(a) };
        }
        pub fn initFallback(a: std.mem.Allocator) Self {
            return .{ .rig = .init(a), .data = .init(a), .launch = &fallback_launch };
        }
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            s.data.deinit();
        }
        pub fn geometry(_: *const Self, sh: Shape) ?metal.Geometry {
            if (sh.rows == 0 or sh.n > 24 or sh.k == 0) return null;
            const tg = kernels.streamGroup(@intCast(sh.k)) orelse return null;
            var g = metal.Geometry{ .grid = .{ ci(sh.k), ci(sh.rows), 1 }, .tg = .{ tg, 1, 1 } };
            g.out(&.{ 1, ci(sh.rows), ci(sh.k) }, .float32);
            g.out(if (msg) &.{1} else &.{ 1, ci(sh.rows), ci(sh.k) }, if (msg) .float32 else dt);
            g.out(if (msg) &.{1} else &.{ 1, ci(sh.rows), ci(sh.k) }, .float32);
            g.out(if (msg) &.{ 1, ci(sh.rows), ci(sh.k) } else &.{1}, .float32);
            g.out(if (msg) &.{ 1, ci(sh.rows), ci(2 * sh.k) } else &.{1}, if (msg) dt else .float32);
            g.unread = if (msg) 0b00110 else 0b11000;
            g.dtype("T", dt);
            g.dtype("XT", dt);
            g.dtype("WT", dt);
            g.dtype("GT", dt);
            g.int("TG", tg);
            g.int("D", sh.k);
            g.int("KT", KT);
            g.int("OFF", 2 * KV);
            g.int("NB", sh.n);
            g.int("BOUND", @intFromBool(boundary));
            g.int("MSG", @intFromBool(msg));
            g.int("SEQ", @intFromBool(sequence));
            return g;
        }
        pub fn prepare(s: *Self, sh: Shape, fill: spec.Fill, seed: u64) !bool {
            if (s.geometry(sh) == null) return false;
            s.shape = sh;
            s.adaptive_probed = false;
            s.last_run_threads = 0;
            s.data.reset(&s.rig);
            var p = std.Random.DefaultPrng.init(seed);
            const rng = p.random();
            const d = sh.k;
            const sw = 2 * KV + 2 * d;
            try s.data.draw(0, rng, sh.rows * d, dt, -2, 2);
            try s.data.draw(1, rng, (if (sequence) @as(usize, 1) else sh.rows) * (KT - 1) * sw, .float32, -1, 1);
            try s.data.draw(2, rng, d * KT, dt, -0.25, 0.25);
            try s.data.draw(3, rng, sh.rows * d, dt, -1, 1);
            try s.data.draw(4, rng, sh.rows * d, .float32, -1, 1);
            try s.data.draw(5, rng, @max(sh.rows * sh.n * d, 1), .float32, -1, 1);
            try s.data.draw(6, rng, d, .float32, -0.02, 0.02);
            try s.data.draw(7, rng, d, dt, 0.5, 1.5);
            try s.data.draw(8, rng, sh.rows * d, .float32, -1, 1);
            s.data.values[9] = try s.data.allocator().dupe(f64, &.{ @as(f32, 1 / @sqrt(@as(f32, KT + 1))), @as(f32, 1e-6) });
            if (fill == .nonfinite) m.host.plant(s.data.values[0], sh.rows, d);
            if (fill == .stress) {
                m.host.intsInto(rng, s.data.values[0], -32, 32);
                @memset(s.data.values[2], 0);
                @memset(s.data.values[6], 0);
                @memset(s.data.values[7], 1);
            }
            try s.data.bind(&s.rig, 0, dt, &.{ 1, ci(sh.rows), ci(d) });
            try s.data.bind(&s.rig, 1, .float32, &.{ if (sequence) 1 else ci(sh.rows), KT - 1, ci(sw) });
            try s.data.bind(&s.rig, 2, dt, &.{ ci(d), KT, 1 });
            try s.data.bind(&s.rig, 3, dt, &.{ 1, ci(sh.rows), ci(d) });
            for ([_]usize{ 4, 8 }) |i| try s.data.bind(&s.rig, i, .float32, &.{ 1, ci(sh.rows), ci(d) });
            try s.data.bind(&s.rig, 5, .float32, if (sh.n == 0) &.{1} else &.{ 1, ci(sh.rows), ci(sh.n), ci(d) });
            try s.data.bind(&s.rig, 6, .float32, &.{ci(d)});
            try s.data.bind(&s.rig, 7, dt, &.{ci(d)});
            try s.data.bind(&s.rig, 9, .float32, &.{2});
            return true;
        }
        pub fn run(s: *Self) !void {
            // The production wrapper probes each instantiation and may narrow the shared stream group.
            // Recompute geometry only after that probe, including when a numerical gate calls run directly.
            if (s.launch.adapts and !s.adaptive_probed) _ = try s.wrapper();
            const g = s.geometry(s.shape).?;
            s.last_run_threads = g.threads();
            try s.rig.launch(s.launch, g, s.data.args[0..10]);
        }
        pub fn count(s: *const Self) usize {
            return (if (msg) 4 else 3) * s.shape.rows * s.shape.k;
        }
        fn outputs(s: *const Self, want: []f64, mag: []f64, calc: m.Calc) !void {
            var arena = std.heap.ArenaAllocator.init(s.rig.a);
            defer arena.deinit();
            const a = arena.allocator();
            const d = s.shape.k;
            const rows = s.shape.rows;
            const nb = s.shape.n;
            const sd = rows * d;
            const ns = nb + @intFromBool(boundary) + 1;
            const sw = 2 * KV + 2 * d;
            const x = s.data.values;
            const terms = try a.alloc(f64, d);
            const res = try a.alloc(V, d);
            const mix = try a.alloc(V, d);
            const logits = try a.alloc(V, ns);
            const source_norm = try a.alloc(V, d);
            for (0..rows) |r| {
                for (res, 0..) |*v, i| {
                    var taps: [KT]f64 = undefined;
                    for (&taps, 0..) |*t, j| {
                        const old = if (sequence) r + j < KT - 1 else j + 1 < KT;
                        const at = if (sequence) r + j else r * (KT - 1) + j;
                        const val = if (old) x[1][at * sw + 2 * KV + i] else x[0][(if (sequence) r + j - (KT - 1) else r) * d + i];
                        t.* = val * x[2][i * KT + j];
                    }
                    const conv = m.round(dt, calc.sum(&taps));
                    v.* = m.round(dt, m.scale(m.add(conv, .exact(x[0][r * d + i])), x[9][0]));
                    if (!boundary) v.* = m.add(v.*, .exact(x[3][r * d + i]));
                    m.emit(want, mag, r * d + i, v.*);
                }
                const rn = try m.norm(calc, a, res, 1e-6);
                for (source_norm, res) |*v, h| v.* = m.mul(h, rn);
                for (logits, 0..) |*l, b| {
                    var inherited: f64 = 0;
                    for (terms, 0..) |*t, i| {
                        const v: V = if (b < nb) .exact(x[5][(r * nb + b) * d + i]) else if (boundary and b == nb) .exact(x[4][r * d + i]) else source_norm[i];
                        t.* = v.v * x[6][i];
                        inherited += v.e * @abs(x[6][i]);
                    }
                    l.* = calc.sum(terms);
                    l.e += inherited;
                }
                const weights = try m.softmax(calc, a, logits);
                for (mix, 0..) |*v, i| {
                    var mixed: [26]V = undefined;
                    for (weights, 0..) |w, b| mixed[b] = m.mul(w, if (b < nb) .exact(x[5][(r * nb + b) * d + i]) else if (boundary and b == nb) .exact(x[4][r * d + i]) else source_norm[i]);
                    v.* = calc.sumValues(mixed[0..weights.len], terms[0..weights.len]);
                }
                const mn = try m.norm(calc, a, mix, 1e-6);
                const ms = try a.alloc(V, d);
                for (ms, 0..) |*v, i| v.* = .exact(x[8][r * d + i]);
                const msn = if (msg) try m.norm(calc, a, ms, 1e-6) else mn;
                for (mix, 0..) |v, i| {
                    if (msg) {
                        m.emit(want, mag, sd + r * d + i, v);
                        m.emit(want, mag, 2 * sd + r * 2 * d + i, m.round(dt, m.mul(v, mn)));
                        m.emit(want, mag, 2 * sd + r * 2 * d + d + i, m.round(dt, m.mul(ms[i], msn)));
                    } else {
                        const out = m.round(dt, m.scale(m.mul(v, mn), x[7][i]));
                        m.emit(want, mag, sd + r * d + i, out);
                        m.emit(want, mag, 2 * sd + r * d + i, out);
                    }
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
            var cfg: config.Config = undefined;
            cfg.hidden_size = @intCast(s.shape.k);
            cfg.sconv_kernel = KT;
            cfg.num_key_value_heads = 1;
            cfg.head_dim = KV;
            cfg.rms_norm_eps = 1e-6;
            const x = s.data.args;
            var l: @import("layers.zig").Layer = undefined;
            l.attn_conv = x[2];
            l.mlp_query = x[6];
            l.post_norm = x[7];
            l.boundary = boundary;
            const o = try kernels.middleStreams(&g, &l, &cfg, x[0], dt, x[3], x[4], if (s.shape.n > 0) x[5] else null, x[1], x[8], dt, msg);
            const bytes = try s.rig.bytesOf(if (msg) &.{ o[0], o[3], o[4] } else &.{ o[0], o[1], o[2] });
            s.adaptive_probed = true;
            return bytes;
        }
    };
}
