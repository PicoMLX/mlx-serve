//! Independent float64 arithmetic for the visit contracts. Errors follow each sum and nonlinear operation,
//! including the dtype roundings between stages; they are forward-error bounds, not measured tolerances.
const std = @import("std");
pub const mlx = @import("../../mlx/ffi.zig");
pub const ops = @import("../../mlx/ops.zig");
pub const kc = @import("../../sdk/testing/kernel_contract/root.zig");
pub const host = kc.host;
pub const spec = kc.spec;
pub const metal = kc.metal;
pub const V = host.Val;
pub const Shape = spec.Shape;

pub fn ci(n: usize) c_int {
    return @intCast(n);
}

pub const Data = struct {
    arena: std.heap.ArenaAllocator,
    values: [14][]f64 = @splat(&.{}),
    args: [14]mlx.mlx_array = undefined,

    pub fn init(a: std.mem.Allocator) Data {
        return .{ .arena = .init(a) };
    }
    pub fn deinit(d: *Data) void {
        d.arena.deinit();
    }
    pub fn reset(d: *Data, rig: *metal.Rig) void {
        rig.reset();
        _ = d.arena.reset(.retain_capacity);
        d.values = @splat(&.{});
    }
    pub fn allocator(d: *Data) std.mem.Allocator {
        return d.arena.allocator();
    }
    pub fn draw(d: *Data, i: usize, rng: std.Random, n: usize, dt: mlx.mlx_dtype, lo: f64, hi: f64) !void {
        d.values[i] = try host.uniform(d.allocator(), rng, n, lo, hi, dt);
    }
    pub fn constant(d: *Data, i: usize, n: usize, v: f64) !void {
        d.values[i] = try d.allocator().alloc(f64, n);
        @memset(d.values[i], v);
    }
    pub fn bind(d: *Data, rig: *metal.Rig, i: usize, dt: mlx.mlx_dtype, shape: []const c_int) !void {
        d.args[i] = try rig.input(dt, shape, d.values[i]);
    }
};

pub const Calc = struct {
    sim: ?struct { acc: spec.Accum, order: spec.Order } = null,

    pub fn sum(c: Calc, x: []const f64) V {
        if (c.sim) |s| return .exact(spec.sumIn(s.acc, s.order, x));
        return .{ .v = host.sum(x), .e = spec.gamma(@floatFromInt(x.len), host.U) * host.sumAbs(x) };
    }

    pub fn sumValues(c: Calc, values: []const V, terms: []f64) V {
        std.debug.assert(values.len == terms.len);
        var inherited: f64 = 0;
        for (values, terms) |v, *t| {
            t.* = v.v;
            inherited += v.e;
        }
        var out = c.sum(terms);
        out.e += inherited;
        return out;
    }
};

pub fn add(a: V, b: V) V {
    const v = a.v + b.v;
    return .{ .v = v, .e = a.e + b.e + host.U * @abs(v) };
}
pub fn mul(a: V, b: V) V {
    var v = host.mul(a, b);
    v.e += host.U * @abs(v.v);
    return v;
}
pub fn scale(a: V, b: f64) V {
    return mul(a, .exact(b));
}
pub fn div(a: V, b: V) V {
    const v = a.v / b.v;
    const den = @max(@abs(b.v) - b.e, std.math.floatMin(f64));
    return .{ .v = v, .e = (a.e + @abs(v) * b.e) / den + 2.5 * 0x1p-23 * @abs(v) };
}
pub fn rsqrt(a: V) V {
    const v = 1 / @sqrt(a.v);
    const low = @max(a.v - a.e, std.math.floatMin(f64));
    return .{ .v = v, .e = a.e / (2 * low * @sqrt(low)) + 4 * 0x1p-23 * @abs(v) };
}
pub fn tanh(a: V) V {
    const v = std.math.tanh(a.v);
    return .{ .v = v, .e = a.e + 4 * 0x1p-23 * @abs(v) };
}
pub fn exp(a: V) V {
    const v = @exp(a.v);
    return .{ .v = v, .e = v * (std.math.expm1(a.e) + host.fastExp(a.v)) };
}
pub fn norm(c: Calc, a: std.mem.Allocator, x: []const V, eps: f64) !V {
    const terms = try a.alloc(f64, x.len);
    var e: f64 = 0;
    for (x, terms) |v, *t| {
        t.* = v.v * v.v;
        e += 2 * @abs(v.v) * v.e + v.e * v.e;
    }
    var ss = c.sum(terms);
    ss.e += e;
    return rsqrt(add(scale(ss, 1 / @as(f64, @floatFromInt(x.len))), .exact(eps)));
}
pub fn softmax(c: Calc, a: std.mem.Allocator, logits: []const V) ![]V {
    const out = try a.alloc(V, logits.len);
    const terms = try a.alloc(f64, logits.len);
    var mx = logits[0];
    for (logits[1..]) |v| if (v.v > mx.v) {
        mx = v;
    };
    var e: f64 = 0;
    for (logits, out, terms) |v, *o, *t| {
        o.* = exp(add(v, .{ .v = -mx.v, .e = mx.e }));
        t.* = o.v;
        e += o.e;
    }
    var den = c.sum(terms);
    den.e += e;
    for (out) |*v| v.* = div(v.*, den);
    return out;
}
pub fn emit(want: []f64, mag: []f64, i: usize, v: V) void {
    want[i] = v.v;
    mag[i] = if (std.math.isFinite(v.e)) host.allowance(v.e) else 0;
}
pub fn round(dt: mlx.mlx_dtype, v: V) V {
    return host.round(dt, v);
}

pub fn noSums(_: Shape) usize {
    return 0;
}
pub fn elements(s: Shape) usize {
    return s.rows * s.k;
}
pub fn three(s: Shape) usize {
    return 3 * elements(s);
}
pub fn four(s: Shape) usize {
    return 4 * elements(s);
}
pub fn six(s: Shape) usize {
    return 6 * elements(s);
}

/// Fixed launches record only their required thread count, a conservative lower bound verified at every shape
/// by the strict offline audit. Adaptive streams assert no wide pipeline cap; their terminal fallback is separate.
pub fn launch(comptime file: [:0]const u8, comptime inputs: []const [*:0]const u8, comptime outputs: []const [*:0]const u8, bytes: u32, adapts: bool) metal.Launch {
    const required: u32 = comptime if (std.mem.eql(u8, file, "prep") or std.mem.eql(u8, file, "downs")) 256 else if (std.mem.eql(u8, file, "relattn")) 128 else if (std.mem.eql(u8, file, "gate") or std.mem.eql(u8, file, "ups")) 64 else if (std.mem.eql(u8, file, "pre") or std.mem.eql(u8, file, "mid")) 0 else @compileError("record this launch's required thread count");
    return .{ .name = "msv_inkling2_" ++ file, .inputs = inputs, .outputs = outputs, .source = @import("../../mlx/kernels.zig").source("plugins/inkling2/kernels/" ++ file ++ ".metal"), .tg_bytes = bytes, .m1 = if (adapts) 0 else required, .m2 = if (adapts) 0 else required, .adapts = adapts };
}
