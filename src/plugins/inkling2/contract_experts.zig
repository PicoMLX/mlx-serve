//! The packed experts, decoded independently on the host: affine widths 2-6/8 at group 64, 4-bit groups 32/128, NVFP4, MXFP4,
//! MXFP8, per-expert scales, reordered picks, prompt outputs and the one-row conv/residual tail.
const std = @import("std");
const m = @import("contract_math.zig");
const kernels = @import("kernels.zig");
const ci = m.ci;
const mlx = m.mlx;
const spec = m.spec;
const metal = m.metal;
const Shape = m.Shape;
const KT = 4;
const CAPS = [4]f64{ 0.25, 4, 0.04, 25 };

pub const Kind = enum { up, down, tail };
fn expertLaunch(comptime up: bool) metal.Launch {
    var l = if (up) m.launch("ups", &.{ "X", "WQ", "WS", "WB", "IDX", "SC", "PR" }, &.{"ACT"}, 0, false) else m.launch("downs", &.{ "ACT", "WQ", "WS", "WB", "IDX", "SC", "WTS", "RES", "ST", "MW", "PR" }, &.{ "HOUT", "XM" }, 8 * 4, false);
    l.header = kernels.bank_header;
    return l;
}
const up_launch = expertLaunch(true);
const down_launch = expertLaunch(false);

const Bank = struct {
    fmt: u32,
    bits: u32,
    gs: usize,
    cols: usize = 0,
    codes: []u8 = &.{},
    scales: []f64 = &.{},
    decoded_scales: []f64 = &.{},
    biases: []f64 = &.{},
    words: []f64 = &.{},
    fn fill(b: *Bank, a: std.mem.Allocator, rng: std.Random, n: usize, k: usize, stress: bool) !void {
        b.cols = k;
        if (b.fmt == 3) {
            var aff = m.host.Affine{ .bits = b.bits, .gs = b.gs, .n = 0, .k = 0 };
            try aff.fill(a, rng, n, k, .bfloat16, stress);
            if (stress) {
                for (0..n) |r| {
                    for (32..40) |j| aff.codes[r * k + j] &= 1;
                    for (0..k / 2) |j| aff.codes[r * k + k - 1 - j] = aff.codes[r * k + j];
                }
                aff.words = try m.host.packAffine(a, aff.codes, n, k, b.bits);
            }
            b.codes = aff.codes;
            b.scales = aff.scales;
            b.biases = aff.biases;
            b.words = aff.words;
            return;
        }
        b.codes = try a.alloc(u8, n * k);
        rng.bytes(b.codes);
        for (b.codes) |*c| c.* = if (b.fmt == 2) (if (stress) 0x38 else m.host.e4m3Code(rng, -3, 1, true)) else c.* & (if (stress) @as(u8, 7) else 15);
        if (stress) for (0..n) |r| {
            for (32..40) |j| if (b.fmt != 2) {
                b.codes[r * k + j] &= 1;
            };
            for (0..k / 2) |j| b.codes[r * k + k - 1 - j] = b.codes[r * k + j];
        };
        b.scales = try a.alloc(f64, n * k / b.gs);
        for (b.scales) |*s| s.* = @floatFromInt(if (b.fmt == 0) (if (stress) @as(u8, 0x38) else m.host.e4m3Code(rng, -4, -1, false)) else if (stress) @as(u8, 127) else @as(u8, 120) + rng.uintLessThan(u8, 4));
        b.decoded_scales = try decodeScales(a, b.fmt, b.scales);
        if (b.fmt == 2) {
            b.words = try a.alloc(f64, b.codes.len / 4);
            for (b.words, 0..) |*w, i| {
                var word: u32 = 0;
                for (0..4) |j| word |= @as(u32, b.codes[4 * i + j]) << @intCast(8 * j);
                w.* = @floatFromInt(word);
            }
        } else b.words = try m.host.packNibbles(a, b.codes);
    }
    fn weight(b: *const Bank, row: usize, col: usize) f64 {
        const at = row * b.cols + col;
        const scale = b.scales[at / b.gs];
        if (b.fmt == 3) return @as(f64, @floatFromInt(b.codes[at])) * scale + b.biases[at / b.gs];
        return (if (b.fmt == 2) m.host.e4m3(b.codes[at]) else m.host.e2m1(b.codes[at])) * b.decoded_scales[at / b.gs];
    }
    fn magnitude(b: *const Bank, row: usize, col: usize) f64 {
        if (b.fmt != 3) return @abs(b.weight(row, col));
        const at = row * b.cols + col;
        return @abs(@as(f64, @floatFromInt(b.codes[at])) * b.scales[at / b.gs]) + @abs(b.biases[at / b.gs]);
    }
};

/// Keep the raw bytes for the device; decode each group once for all host dot products and precision simulations.
fn decodeScales(a: std.mem.Allocator, fmt: u32, encoded: []const f64) ![]f64 {
    const decoded = try a.alloc(f64, encoded.len);
    for (decoded, encoded) |*out, byte| out.* = if (fmt == 0) m.host.e4m3(@intFromFloat(byte)) else m.host.e8m0(@intFromFloat(byte));
    return decoded;
}

pub fn Packed(comptime kind: Kind, comptime fmt: u32, comptime bits: u32, comptime gs: usize, comptime sequences: usize, comptime scaled: bool) type {
    return struct {
        const Self = @This();
        const up = kind == .up;
        const tail = kind == .tail;
        rig: metal.Rig,
        launch: *const metal.Launch = if (up) &up_launch else &down_launch,
        data: m.Data,
        bank: Bank = .{ .fmt = fmt, .bits = bits, .gs = gs },
        shape: Shape = undefined,
        pub fn init(a: std.mem.Allocator) Self {
            return .{ .rig = .init(a), .data = .init(a) };
        }
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            s.data.deinit();
        }
        fn picks(sh: Shape) usize {
            return @min(sh.rows, 8);
        }
        pub fn geometry(_: *const Self, sh: Shape) ?metal.Geometry {
            if (sh.rows == 0 or sh.n == 0 or sh.n % 2 != 0 or sh.k == 0 or sh.k % gs != 0 or sh.k % 32 != 0) return null;
            const ksel = picks(sh);
            var g = metal.Geometry{ .grid = if (up) .{ 32, ci(sh.n), ci(sequences * ksel) } else .{ 32, ci(ksel), ci(sequences * sh.n) }, .tg = if (up) .{ 32, 2, 1 } else .{ 32, ci(ksel), 1 } };
            if (up) g.out(&.{ ci(sequences * ksel), 1, ci(sh.n) }, .bfloat16) else {
                g.out(if (tail) &.{ 1, ci(sequences), ci(sh.n) } else &.{1}, if (tail) .bfloat16 else .float32);
                g.out(&.{ 1, ci(sequences), ci(sh.n) }, .bfloat16);
                if (!tail) g.unread = 1;
            }
            g.dtype("T", .bfloat16);
            if (!up) g.dtype("WT", .bfloat16);
            if (up) {
                g.int("K", sh.k);
                g.int("I", sh.n);
            } else {
                g.int("I", sh.k);
                g.int("D", sh.n);
            }
            g.int("GS", gs);
            g.int("FMT", fmt);
            g.int("ROWS", 1);
            g.int("BITS", bits);
            g.dtype("S", if (fmt == 3) .bfloat16 else .uint8);
            g.int("SCALED", @intFromBool(scaled));
            g.int("KSEL", ksel);
            if (!up) {
                g.int("KT", KT);
                g.int("TAIL", @intFromBool(tail));
            }
            return g;
        }
        pub fn prepare(s: *Self, sh: Shape, fill: spec.Fill, seed: u64) !bool {
            if (s.geometry(sh) == null) return false;
            s.shape = sh;
            s.data.reset(&s.rig);
            var p = std.Random.DefaultPrng.init(seed);
            const rng = p.random();
            const ksel = picks(sh);
            const nr = if (up) 2 * sh.n else sh.n;
            const a = s.data.allocator();
            s.bank = .{ .fmt = fmt, .bits = bits, .gs = gs };
            try s.bank.fill(a, rng, sh.rows * nr, sh.k, fill == .stress);
            try s.data.draw(0, rng, sequences * (if (up) @as(usize, 1) else ksel) * sh.k, .bfloat16, -1, 1);
            s.data.values[1] = s.bank.words;
            s.data.values[2] = s.bank.scales;
            s.data.values[3] = if (fmt == 3) s.bank.biases else s.bank.scales;
            try s.data.constant(4, sequences * ksel, 0);
            for (s.data.values[4], 0..) |*v, i| v.* = @floatFromInt((3 * i + i / ksel) % sh.rows);
            try s.data.draw(5, rng, sh.rows, .float32, 0.5, 1.5);
            if (up) s.data.values[6] = try a.dupe(f64, &CAPS) else {
                try s.data.draw(6, rng, sequences * ksel, .float32, 0.1 / @as(f64, @floatFromInt(ksel)), 1.0 / @as(f64, @floatFromInt(ksel)));
                try s.data.draw(7, rng, sequences * sh.n, .float32, -1, 1);
                try s.data.draw(8, rng, (KT - 1) * (64 + sh.n), .float32, -1, 1);
                try s.data.draw(9, rng, sh.n * KT, .bfloat16, -0.25, 0.25);
                try s.data.constant(10, 1, @as(f32, 1.0 / @sqrt(@as(f32, KT + 1))));
            }
            if (fill == .nonfinite) {
                if (tail) {
                    const pattern = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) };
                    @memcpy(s.data.values[7][0..@min(pattern.len, sh.n)], pattern[0..@min(pattern.len, sh.n)]);
                    if (sh.n > 3) {
                        // Column3 adds opposite infinities at the residual seam, independently of the NaN column.
                        s.data.values[8][64 + 3] = -std.math.inf(f64);
                        s.data.values[9][3 * KT] = 1;
                    }
                } else if (sequences > 1) m.host.plant(s.data.values[0], @min(4, sequences), sh.k) else if (scaled) s.data.values[5][0] = std.math.nan(f64);
            }
            if (fill == .stress) {
                // Reflected weights and reversed negative halves: fp32 sums stay exact below 2^24, while
                // half/bf16 lose the eight small offsets even when summing a pairwise or SIMD-strided tree.
                m.host.mirroredWide(rng, s.data.values[0], sh.k);
                for (0..s.data.values[0].len / sh.k) |r| std.mem.reverse(f64, s.data.values[0][r * sh.k + sh.k / 2 ..][0 .. sh.k / 2]);
                @memset(s.data.values[5], 1);
                if (!up) {
                    @memset(s.data.values[6], 1);
                    @memset(s.data.values[9], 0);
                }
            }
            try s.data.bind(&s.rig, 0, .bfloat16, if (up) &.{ 1, ci(sequences), ci(sh.k) } else &.{ ci(sequences * ksel), 1, ci(sh.k) });
            try s.data.bind(&s.rig, 1, .uint32, &.{ ci(sh.rows), ci(nr), ci(sh.k * bits / 32) });
            try s.data.bind(&s.rig, 2, if (fmt == 3) .bfloat16 else .uint8, &.{ ci(sh.rows), ci(nr), ci(sh.k / gs) });
            try s.data.bind(&s.rig, 3, if (fmt == 3) .bfloat16 else .uint8, &.{ ci(sh.rows), ci(nr), ci(sh.k / gs) });
            try s.data.bind(&s.rig, 4, .uint32, &.{ci(sequences * ksel)});
            try s.data.bind(&s.rig, 5, .float32, &.{ci(sh.rows)});
            if (up) try s.data.bind(&s.rig, 6, .float32, &.{4}) else {
                try s.data.bind(&s.rig, 6, .float32, &.{ 1, ci(sequences), ci(ksel) });
                try s.data.bind(&s.rig, 7, .float32, &.{ 1, ci(sequences), ci(sh.n) });
                try s.data.bind(&s.rig, 8, .float32, &.{ 1, KT - 1, ci(64 + sh.n) });
                try s.data.bind(&s.rig, 9, .bfloat16, &.{ ci(sh.n), KT, 1 });
                try s.data.bind(&s.rig, 10, .float32, &.{1});
            }
            return true;
        }
        pub fn run(s: *Self) !void {
            try s.rig.launch(s.launch, s.geometry(s.shape).?, s.data.args[0..if (up) 7 else 11]);
        }
        pub fn count(s: *const Self) usize {
            return sequences * s.shape.n * (if (up) picks(s.shape) else if (tail) @as(usize, 2) else 1);
        }
        fn dot(s: *const Self, calc: m.Calc, row: usize, x: []const f64, terms: []f64) m.V {
            var magnitude: f64 = 0;
            for (terms, x, 0..) |*t, v, i| {
                t.* = v * s.bank.weight(row, i);
                magnitude += @abs(v) * s.bank.magnitude(row, i);
            }
            var v: m.V = undefined;
            if (fmt == 3) {
                var inherited: f64 = 0;
                for (0..x.len / 16) |chunk| {
                    var qx: [16]f64 = undefined;
                    const at = chunk * 16;
                    for (&qx, x[at..][0..16], 0..) |*t, val, j| t.* = val * @as(f64, @floatFromInt(s.bank.codes[row * x.len + at + j]));
                    const sx = calc.sum(x[at..][0..16]);
                    const sq = calc.sum(&qx);
                    const group = (row * x.len + at) / gs;
                    const qterm = m.scale(sq, s.bank.scales[group]);
                    const bterm = m.scale(sx, s.bank.biases[group]);
                    var part = calc.sum(&.{ qterm.v, bterm.v });
                    part.e += qterm.e + bterm.e;
                    terms[chunk] = part.v;
                    inherited += part.e;
                }
                // The shipped reader and MLX qdot factor each16-value chunk. Preserve its Inf/NaN classes;
                // distributing an infinite x over predecoded (q*s+b) weights is not the same formula.
                v = calc.sum(terms[0 .. x.len / 16]);
                v.e += inherited;
            } else v = calc.sum(terms);
            // Affine chunks factor the scale/bias after the integer-code dots: include both terms' magnitudes.
            v.e += spec.gamma(@floatFromInt(2 * x.len + 4), m.host.U) * magnitude;
            return m.round(.bfloat16, v);
        }
        fn outputs(s: *const Self, want: []f64, mag: []f64, calc: m.Calc) !void {
            const sh = s.shape;
            const x = s.data.values;
            const ksel = picks(sh);
            const terms = try s.rig.a.alloc(f64, sh.k);
            defer s.rig.a.free(terms);
            for (0..sequences) |seq| for (0..sh.n) |r| {
                if (up) {
                    for (0..ksel) |pick| {
                        const p = seq * ksel + pick;
                        const ex: usize = @intFromFloat(x[4][p]);
                        const rowx = x[0][seq * sh.k ..][0..sh.k];
                        const scale: f64 = if (scaled) x[5][ex] else 1;
                        const g = m.round(.float32, m.scale(s.dot(calc, (ex * 2 * sh.n) + 2 * r, rowx, terms), scale));
                        const u = m.round(.float32, m.scale(s.dot(calc, (ex * 2 * sh.n) + 2 * r + 1, rowx, terms), scale));
                        const cap = m.mul(m.scale(m.tanh(m.scale(g, CAPS[0])), CAPS[1]), m.scale(m.tanh(m.scale(u, CAPS[2])), CAPS[3]));
                        // exp(-g) is a stored float32 intermediate (as in MLX Sigmoid), including overflow to
                        // infinity. A float64 exp here would invent nonzero outputs after the real denominator overflows.
                        const exponent = m.round(.float32, m.exp(.{ .v = -g.v, .e = g.e }));
                        const val = m.div(cap, m.add(.exact(1), exponent));
                        m.emit(want, mag, p * sh.n + r, val);
                    }
                } else {
                    var parts: [8]f64 = undefined;
                    var inherited: f64 = 0;
                    for (0..ksel) |pick| {
                        const p = seq * ksel + pick;
                        const ex: usize = @intFromFloat(x[4][p]);
                        var v = s.dot(calc, ex * sh.n + r, x[0][p * sh.k ..][0..sh.k], terms);
                        if (scaled) v = m.scale(v, x[5][ex]);
                        v = m.scale(v, x[6][p]);
                        parts[pick] = v.v;
                        inherited += v.e;
                    }
                    var xm = calc.sum(parts[0..ksel]);
                    xm.e += inherited;
                    xm = m.round(.bfloat16, xm);
                    const off = seq * sh.n + r;
                    m.emit(want, mag, (if (tail) sequences * sh.n else 0) + off, xm);
                    if (tail) {
                        var taps: [KT]f64 = undefined;
                        var e: f64 = 0;
                        for (&taps, 0..) |*t, j| {
                            const v: m.V = if (j + 1 == KT) xm else .exact(x[8][j * (64 + sh.n) + 64 + r]);
                            const weight = x[9][r * KT + j];
                            t.* = v.v * weight;
                            e += @abs(weight) * v.e;
                        }
                        var conv = calc.sum(&taps);
                        conv.e += e;
                        conv = m.round(.bfloat16, conv);
                        const val = m.add(.exact(x[7][r]), m.round(.bfloat16, m.scale(m.add(conv, xm), x[10][0])));
                        m.emit(want, mag, off, val);
                    }
                }
            };
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
            const bank = .{ .w = x[1], .s = x[2], .b = x[3], .qp = .{ .bits = bits, .group_size = gs } };
            if (up) return s.rig.bytesOf(&.{try kernels.packedUp(&g, &bank, x[0], x[4], x[5], x[6], ci(picks(s.shape)), fmt, scaled)});
            const o = try kernels.packedDown(&g, &bank, x[0], x[4], x[5], x[6], x[7], x[8], x[9], x[10], ci(picks(s.shape)), fmt, scaled, tail);
            return s.rig.bytesOf(if (tail) &o else &.{o[1]});
        }
    };
}

test "kernel contract inkling2: cached group scales preserve every encoded scale byte" {
    var encoded: [256]f64 = undefined;
    for (&encoded, 0..) |*v, code| v.* = @floatFromInt(code);
    for ([_]u32{ 0, 1, 2 }) |fmt| {
        const decoded = try decodeScales(std.testing.allocator, fmt, &encoded);
        defer std.testing.allocator.free(decoded);
        for (decoded, 0..) |v, code| {
            // The NV scale reader places these bits in a half (leaving2^-8 out); MX uses a biased exponent.
            const half_bits: u16 = @intCast(((code & 0x7f) << 7) | ((code & 0x80) << 8));
            const expected: f64 = if (fmt == 0) @as(f64, @as(f16, @bitCast(half_bits))) * 256 else std.math.ldexp(@as(f64, 1), @as(i32, @intCast(code)) - 127);
            try std.testing.expectEqual(expected, v);
            try std.testing.expect(std.math.isFinite(v));
            try std.testing.expectEqual(@as(f64, @floatFromInt(code)), encoded[code]);
        }
    }
}
