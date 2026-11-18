//! The MLX text-generation engine behind libPicoMlxServe.dylib (`src/embed_lib.zig`): open a model directory, turn
//! text into token ids and back, and generate from a prompt of token ids. It is the server's own boot
//! (`main.zig`, `host/http/server.zig` `serve`) without the HTTP listener: one scheduler, whose inference thread
//! loads the model and runs every forward, and one registry holding that model.
//!
//! One model is open at a time: the engine's launch state (`server.launch()`, the admission hooks) is one per
//! process, and MLX has one caller, the inference thread.
//!
//! What opening a model leaves alone: no signal handler (the `Io` is `init_single_threaded`'s, which installs
//! none), no environment variable, no listener, no outside plugin, no sleep assertion, and nothing written outside
//! MLX and Metal's own doing: the warm-up marker, the SSD prefix cache and the round-cost table stay off, and
//! `model-settings.json` is not read (`embed_lib.mlx_serve_embedded`). It does set this MLX's buffer-pool cap and
//! wired limit, as the server does (`configureProcess`, `mlx.applyWiredPolicy`).

const std = @import("std");
const lib = @import("embed_lib.zig");
const abi = @import("abi/engine.zig");
const mlx = @import("mlx/ffi.zig");
const mlx_state = @import("mlx/state.zig");
const log = @import("sdk/util/log.zig");
const model_mod = @import("sdk/config/model.zig");
const tokenizer_mod = @import("sdk/tokenizer/tokenizer.zig");
const chat_mod = @import("host/chat/chat.zig");
const gen_mod = @import("host/media_jobs/gen.zig");
const registry_mod = @import("host/registry/registry.zig");
const requests = @import("host/gpu_thread/requests.zig");
const server_mod = @import("host/http/server.zig");
const scheduler = @import("engines/mlx/scheduler.zig");
const configure = @import("engines/mlx/configure.zig");
const admission = @import("engines/mlx/engine_admission.zig");
const admit_mod = @import("engines/mlx/admission/admit.zig");
const preflight = @import("engines/mlx/runtime/preflight.zig");
const publishHotCacheResidency = @import("engines/mlx/cache/budget.zig").publishHotCacheResidency;
const Slot = @import("engines/mlx/slot.zig").Slot;
const Scheduler = scheduler.Scheduler;
const LoadedModel = registry_mod.LoadedModel;
const ModelRegistry = registry_mod.ModelRegistry;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;

const gpa = std.heap.c_allocator;

/// Requests decoded in one batched pass; more wait their turn.
const max_concurrent: u32 = 8;

/// `std.Io.Threaded.init` installs SIGIO and SIGPIPE handlers; the statically initialized instance installs none.
/// Nothing here cancels a task or runs one concurrently, which is all that instance leaves out.
var threaded: std.Io.Threaded = blk: {
    var t: std.Io.Threaded = .init_single_threaded;
    t.allocator = gpa;
    break :blk t;
};

fn theIo() std.Io {
    return threaded.io();
}

/// `f(args)` on a thread of the library's own, started for the call and joined before it returns; null when the
/// thread could not start. A pack's config parse alone takes over 1 MB of stack, and a caller may have 512 KB (a
/// Grand Central Dispatch or Swift concurrency thread): what parses a config, or loads or frees a model, runs here.
/// The calls left on the caller's thread stay under 64 KB.
fn onOwnStack(comptime R: type, comptime f: anytype, args: anytype) ?R {
    const Call = struct {
        args: @TypeOf(args),
        result: R = undefined,

        fn run(c: *@This()) void {
            c.result = @call(.auto, f, c.args);
        }
    };
    var call: Call = .{ .args = args };
    const t = std.Thread.spawn(.{}, Call.run, .{&call}) catch return null;
    t.join();
    return call.result;
}

/// Guards `open_engine` and the once-per-process settings.
var open_mutex: @import("sdk/util/os_mutex.zig").Mutex = .{};
var open_engine: ?*Engine = null;
var process_configured = false;

pub const EventCb = *const fn (ctx: ?*anyopaque, tokens: ?[*]const i32, n: i32, status: i32, usage: ?*const [6]i64) callconv(.c) void;

const Status = enum(i32) { stopped = 0, cancelled = 1, failed = 2, length = 3 };

const Request = struct {
    engine: *Engine,
    id: u64,
    cb: EventCb,
    ctx: ?*anyopaque,
    /// Set by `submit` once the scheduler has the slot, or by the slot's last event, whichever comes first.
    slot: ?*Slot = null,
    /// The thread inside `submit`. Events come from the inference thread; one from this thread is the scheduler
    /// freeing a slot it refused, never a real one.
    submitter: std.Thread.Id,
    cancel_wanted: bool = false,
    prompt_tokens: u32 = 0,
    cached: u32 = 0,
    generated: i64 = 0,
    /// An end token the engine handed over as a generated token: the reply ends there.
    end_token: i64 = -1,
};

const Engine = struct {
    io: std.Io,
    model_dir: []u8,
    registry: *ModelRegistry,
    entry: *LoadedModel,
    sch: *Scheduler,
    /// Prompt and reply together, frozen at load.
    context: u32,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    /// Accepted requests that have not had their final callback.
    live: std.ArrayList(*Request) = .empty,
    /// Requests past their final callback whose slot the reaper still hands back.
    done: std.ArrayList(*Request) = .empty,
    reaping: u32 = 0,
    stop: bool = false,
    reaper: ?std.Thread = null,

    fn config(e: *const Engine) *ModelConfig {
        return e.entry.config.?;
    }

    fn tok(e: *const Engine) *const Tokenizer {
        return e.entry.tokenizer.?;
    }

    fn findLive(e: *Engine, id: u64) ?*Request {
        for (e.live.items) |r| if (r.id == id) return r;
        return null;
    }
};

// ── sink: the scheduler's events for one request, on the inference thread

fn isEnd(e: *const Engine, token: u32) bool {
    for (e.config().eosTokenSlice()) |id| if (id == token) return true;
    return false;
}

fn onEvent(ctx: *anyopaque, slot_addr: u64, ev: *const abi.Event) callconv(.c) abi.Action {
    const r: *Request = @ptrCast(@alignCast(ctx));
    if (std.Thread.getCurrentId() == r.submitter) return .go;
    const e = r.engine;
    switch (ev.kind) {
        .prefilled => {
            r.prompt_tokens = ev.prompt_tokens;
            r.cached = ev.cached;
        },
        .tokens => {
            if (r.end_token >= 0) return .stop;
            const ids = abi.items(u32, ev.tokens);
            // The engine ends a reply before its end token; one that still arrives as a token ends it here.
            const n = for (ids, 0..) |t, i| {
                if (isEnd(e, t)) break i;
            } else ids.len;
            if (n > 0) {
                r.generated += @intCast(n);
                r.cb(r.ctx, @ptrCast(ids.ptr), @intCast(n), -1, null);
            }
            if (n < ids.len) {
                r.end_token = ids[n];
                return .stop;
            }
        },
        .audio => {},
        .finished => {
            const slot: *Slot = @ptrFromInt(slot_addr);
            var status: Status = switch (ev.reason) {
                .stop => .stopped,
                .length => .length,
                .cancelled => .cancelled,
                .failed => .failed,
            };
            var end: i64 = -1;
            if (r.end_token >= 0) {
                status = .stopped;
                end = r.end_token;
            } else if (status == .stopped) {
                // The generator stopped on the token it was about to yield.
                if (slot.legacy_gen) |*g| if (isEnd(e, g.next_token_id)) {
                    end = g.next_token_id;
                };
            }
            if (status == .failed) log.err("pico_ms: request {d} failed: {s}\n", .{ r.id, ev.verdict.why.slice() });
            const stats: abi.Stats = if (ev.stats) |s| s.* else .{};
            const usage: [6]i64 = .{
                r.prompt_tokens -| r.cached,
                r.cached,
                r.generated,
                @intCast(stats.prefill_ns / std.time.ns_per_us),
                @intCast(stats.decode_ns / std.time.ns_per_us),
                end,
            };
            r.cb(r.ctx, null, 0, @backingInt(status), &usage);
            // Past its final callback: the reaper hands the slot back to the scheduler.
            e.mu.lockUncancelable(e.io);
            defer e.mu.unlock(e.io);
            r.slot = slot;
            for (e.live.items, 0..) |x, i| if (x == r) {
                _ = e.live.swapRemove(i);
                break;
            };
            e.done.append(gpa, r) catch @panic("pico_ms: out of memory");
            e.cond.broadcast(e.io);
        },
    }
    return .go;
}

/// Hands each finished request's slot back to the scheduler. `complete` waits out the pass that holds the slot, so
/// it cannot run on the inference thread that just finished it.
fn reaperLoop(e: *Engine) void {
    e.mu.lockUncancelable(e.io);
    defer e.mu.unlock(e.io);
    while (true) {
        while (e.done.items.len == 0 and !e.stop) e.cond.waitUncancelable(e.io, &e.mu);
        const r = e.done.pop() orelse return;
        e.reaping += 1;
        e.mu.unlock(e.io);
        scheduler.complete(e.sch, r.slot.?);
        gpa.destroy(r);
        e.mu.lockUncancelable(e.io);
        e.reaping -= 1;
        e.cond.broadcast(e.io);
    }
}

// ── model directory checks, without a handle

fn dirOf(model_dir: ?[*:0]const u8) ?[]const u8 {
    const d = std.mem.span(model_dir orelse return null);
    return if (std.fs.path.isAbsolute(d)) std.mem.trimEnd(u8, d, "/") else null;
}

fn hasFile(io: std.Io, dir: []const u8, name: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name }) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// The config of a pack this library's MLX engine generates text from; caller deinits. An error otherwise.
fn servedConfig(io: std.Io, dir: []const u8) !ModelConfig {
    var config = try model_mod.parseConfig(io, gpa, dir);
    errdefer config.deinit(gpa);
    // Only a plugin of the tree serves a pack (`load.zig`); a media pack is no text model.
    const binding = config.plugin orelse return error.UnsupportedModel;
    if (binding.outside or gen_mod.modalityFromType(config.model_type) != null) return error.UnsupportedModel;
    if (binding.parsed.caps.embed or binding.parsed.caps.companion or binding.parsed.caps.drafter) return error.UnsupportedModel;
    if (!hasFile(io, dir, "tokenizer.json") and !hasFile(io, dir, "vocab.json")) return error.TokenizerMissing;
    return config;
}

/// tokenizer_config.json's `eos_token`, the token a chat turn ends on, in `gpa`; null when the file names none.
fn turnEndToken(io: std.Io, dir: []const u8) ?[]const u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/tokenizer_config.json", .{dir}) catch return null;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20)) catch return null;
    defer gpa.free(text);
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get("eos_token") orelse return null;
    const name = switch (v) {
        .string => |s| s,
        .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else return null) else return null,
        else => return null,
    };
    return gpa.dupe(u8, name) catch null;
}

fn loadBytes(io: std.Io, dir: []const u8) !u64 {
    var config = try servedConfig(io, dir);
    defer config.deinit(gpa);
    const bytes = try preflight.residentModelDiskBytes(gpa, io, dir, &config);
    return if (bytes == 0) error.WeightsMissing else bytes;
}

/// The bytes loading the pack at `dir` takes, -1 when this library does not serve it.
fn loadBytesOf(dir: []const u8) i64 {
    return @intCast(loadBytes(theIo(), dir) catch return -1);
}

export fn pico_ms_can_serve(model_dir: ?[*:0]const u8) i32 {
    return @intFromBool(pico_ms_load_bytes(model_dir) >= 0);
}

export fn pico_ms_load_bytes(model_dir: ?[*:0]const u8) i64 {
    lib.ensureInit();
    const dir = dirOf(model_dir) orelse return -1;
    return onOwnStack(i64, loadBytesOf, .{dir}) orelse -1;
}

// ── open, close

/// The engine's process-wide settings, as `main` sets them from its flags, once. Sleep is not held off while a
/// request runs, and MLX's buffer pool is capped (its own default is most of RAM).
fn configureProcess() void {
    if (process_configured) return;
    process_configured = true;
    configure.apply(&.{ .no_vision = true, .prevent_sleep = false });
    mlx_state.applyMlxCacheLimit();
    var metal = false;
    if (mlx.mlx_metal_is_available(&metal) == 0 and metal) {
        const gpu = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu);
        _ = mlx.mlx_set_default_device(gpu);
    }
    _ = mlx.mlx_random_seed(@intCast(std.Io.Timestamp.now(theIo(), .real).toMilliseconds()));
}

fn fail(err: ?[*]u8, err_len: i32, comptime fmt: []const u8, args: anytype) ?*anyopaque {
    if (err) |p| if (err_len > 0) {
        const buf = p[0..@intCast(err_len)];
        const text = std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args) catch buf[0 .. buf.len - 1];
        buf[text.len] = 0;
    };
    return null;
}

export fn pico_ms_open(model_dir: ?[*:0]const u8, err: ?[*]u8, err_len: i32) ?*anyopaque {
    lib.ensureInit();
    const dir = dirOf(model_dir) orelse return fail(err, err_len, "the model directory must be an absolute path", .{});
    return onOwnStack(?*anyopaque, openHandle, .{ dir, err, err_len }) orelse fail(err, err_len, "no thread to load the model on", .{});
}

fn openHandle(dir: []const u8, err: ?[*]u8, err_len: i32) ?*anyopaque {
    open_mutex.lock();
    defer open_mutex.unlock();
    if (open_engine != null) return fail(err, err_len, "a model is already open: this library serves one at a time", .{});
    configureProcess();
    const e = open(dir) catch |x| {
        var mlx_buf: [512]u8 = undefined;
        const why = mlx.takeError(&mlx_buf) orelse "";
        return fail(err, err_len, "{s}{s}{s}", .{ @errorName(x), if (why.len > 0) ": " else "", why });
    };
    open_engine = e;
    return e;
}

fn open(dir: []const u8) !*Engine {
    const io = theIo();
    const config = try gpa.create(ModelConfig);
    var owned = true; // until the load installs config, tokenizer and chat config on the registry's entry
    config.* = servedConfig(io, dir) catch |x| {
        gpa.destroy(config);
        return x;
    };
    defer if (owned) {
        config.deinit(gpa);
        gpa.destroy(config);
    };
    const tok = try gpa.create(Tokenizer);
    tok.* = tokenizer_mod.loadTokenizerAny(io, gpa, dir) catch |x| {
        gpa.destroy(tok);
        return x;
    };
    defer if (owned) {
        tok.deinit();
        gpa.destroy(tok);
    };
    // No chat template: the caller renders its own prompts, and loading one would bring in the Jinja renderer. The
    // turn terminator it names (Qwen's `<|im_end|>`) still joins the end tokens below.
    const chat_config = try gpa.create(chat_mod.ChatConfig);
    chat_config.* = .{
        .chat_template = gpa.dupe(u8, "") catch |x| {
            gpa.destroy(chat_config);
            return x;
        },
        .bos_token = null,
        .eos_token = turnEndToken(io, dir),
        .add_bos_token = false,
        .allocator = gpa,
        .renders_tool_turn = false,
    };
    defer if (owned) {
        chat_config.deinit();
        gpa.destroy(chat_config);
    };
    // No drafting head and no Neural Engine, whatever the pack ships: the reply is the plain decode.
    config.mtp_override = false;
    config.ane_override = false;
    config.applyTokenizer(tok, chat_config.eos_token);

    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    const model_dir = try gpa.dupe(u8, dir);
    errdefer gpa.free(model_dir);
    const registry = try ModelRegistry.init(gpa, io, null, 1, 0, null);
    errdefer registry.deinit();
    const entry = try registry.registerStub(std.fs.path.basename(model_dir), model_dir, null);
    try registry.setDefault(entry.id);
    // Whatever happens next, the entry owns these three if the load got as far as installing them.
    defer owned = entry.config == null;

    const l = server_mod.launch();
    l.configured_kv_quant = .dense;
    defer l.configured_kv_quant = null;
    const sch = scheduler.init(gpa, io, .{
        .registry = registry,
        .entry = entry,
        .config = config,
        .tok = tok,
        .chat_config = chat_config,
        .model_dir = model_dir,
        .no_drafter = true,
        .mtp_enabled = false,
        .load_vision = false,
        // The load's own warm-up leaves a marker in the user's cache directory and serves requests of its own while
        // idle; `warmUp` below runs its forwards alone.
        .warmup_eager = false,
        .launch_warmup_eager = false,
        .prefix_cache_capacity = l.prefix_cache_capacity,
        .prefix_cache_ram_enabled = l.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = l.prefix_cache_mem_bytes,
        .prefix_cache_clamp = true,
        .prefix_cache_disk_bytes = 0,
        .ssm_checkpoint_stride = l.ssm_checkpoint_stride,
        .ssm_checkpoint_max = l.ssm_checkpoint_max,
        .tokenize_cache_entries = 0,
    }, max_concurrent) catch |x| {
        if (entry.error_name) |name| log.err("pico_ms: {s} not loaded: {s}\n", .{ model_dir, name });
        return if (entry.error_name) |name| loadError(name) else x;
    };
    errdefer scheduler.deinit(sch);
    l.global_scheduler = sch;
    errdefer l.global_scheduler = null;
    admit_mod.installSchedulerHooks(true);
    errdefer admit_mod.installSchedulerHooks(false);

    // The weights are bound lazily: forward once so they are resident and the decode kernels compiled before the
    // context is sized against what memory is left, and before the first request.
    var warm: requests.GenRequest = .{ .ctx = entry, .run = warmUp, .model = entry };
    try scheduler.runGeneration(sch, &warm);

    e.* = .{
        .io = io,
        .model_dir = model_dir,
        .registry = registry,
        .entry = entry,
        .sch = sch,
        .context = admission.table.pin_context(entry.config.?, false),
    };
    e.reaper = try std.Thread.spawn(.{}, reaperLoop, .{e});
    return e;
}

/// On the inference thread: the two forwards of the server's eager warm-up (`load.zig`), without its marker file.
fn warmUp(ctx: *anyopaque) void {
    const entry: *LoadedModel = @ptrCast(@alignCast(ctx));
    const xfm = entry.transformer orelse return;
    xfm.warmup(0) catch |x| {
        var buf: [256]u8 = undefined;
        log.warn("pico_ms: warm-up failed ({s}: {s}); the first request loads the weights itself\n", .{ @errorName(x), mlx.takeError(&buf) orelse "" });
    };
}

/// The load's own refusals by name; anything else is `LoadFailed`, with the name in the log.
fn loadError(name: []const u8) anyerror {
    inline for (.{ error.InsufficientMemory, error.PluginDeclinedNoLegacy, error.OutOfMemory, error.FileNotFound }) |x| {
        if (std.mem.eql(u8, name, @errorName(x))) return x;
    }
    return error.LoadFailed;
}

export fn pico_ms_close(h: ?*anyopaque) void {
    const e: *Engine = @ptrCast(@alignCast(h orelse return));
    // Without a thread of its own the model is still closed, on the caller's.
    onOwnStack(void, closeHandle, .{e}) orelse closeHandle(e);
}

fn closeHandle(e: *Engine) void {
    const io = e.io;
    {
        e.mu.lockUncancelable(io);
        defer e.mu.unlock(io);
        for (e.live.items) |r| cancelLocked(r);
        // Every request has had its final callback and its slot is back with the scheduler.
        while (e.live.items.len > 0 or e.done.items.len > 0 or e.reaping > 0) e.cond.waitUncancelable(io, &e.mu);
        e.stop = true;
        e.cond.broadcast(io);
    }
    if (e.reaper) |t| t.join();

    open_mutex.lock();
    defer open_mutex.unlock();
    admit_mod.installSchedulerHooks(false);
    server_mod.launch().global_scheduler = null;
    scheduler.deinit(e.sch);
    e.registry.deinit();
    e.live.deinit(gpa);
    e.done.deinit(gpa);
    gpa.free(e.model_dir);
    gpa.destroy(e);
    open_engine = null;
    // The model's buffers are in MLX's pool now: give them back, and wire nothing.
    _ = mlx.mlx_clear_cache();
    var prev: usize = 0;
    _ = mlx.mlx_set_wired_limit(&prev, 0);
}

// ── tokens

export fn pico_ms_tokenize(h: ?*anyopaque, text: ?[*]const u8, len: i32, out: ?[*]i32, cap: i32) i32 {
    const e: *Engine = @ptrCast(@alignCast(h orelse return -1));
    if (len < 0 or (len > 0 and text == null)) return -1;
    const bytes: []const u8 = if (len > 0) text.?[0..@intCast(len)] else "";
    if (!std.unicode.utf8ValidateSlice(bytes)) return -1;
    const ids = e.tok().encode(gpa, bytes) catch return -1;
    defer gpa.free(ids);
    if (ids.len > std.math.maxInt(i32)) return -1;
    if (out) |o| if (cap > 0) {
        const n = @min(ids.len, @as(usize, @intCast(cap)));
        for (o[0..n], ids[0..n]) |*dst, id| dst.* = @bitCast(id);
    };
    return @intCast(ids.len);
}

export fn pico_ms_detokenize(h: ?*anyopaque, tokens: ?[*]const i32, n: i32, out: ?[*]u8, cap: i32) i32 {
    const e: *Engine = @ptrCast(@alignCast(h orelse return -1));
    if (n < 0 or (n > 0 and tokens == null)) return -1;
    const ids: []const u32 = if (n > 0) @ptrCast(tokens.?[0..@intCast(n)]) else &.{};
    for (ids) |id| if (id > std.math.maxInt(i32)) return -1;
    const text = e.tok().decode(gpa, ids, false) catch return -1;
    defer gpa.free(text);
    if (text.len > std.math.maxInt(i32)) return -1;
    if (out) |o| if (cap > 0) {
        const m = @min(text.len, @as(usize, @intCast(cap - 1)));
        @memcpy(o[0..m], text[0..m]);
        o[m] = 0;
    };
    return @intCast(text.len);
}

// ── requests

export fn pico_ms_submit(h: ?*anyopaque, id: u64, prompt: ?[*]const i32, n: i32, max_tokens: i32, temperature: f32, top_p: f32, cb: ?EventCb, ctx: ?*anyopaque) i32 {
    const e: *Engine = @ptrCast(@alignCast(h orelse return 1));
    const callback = cb orelse return 1;
    if (n <= 0 or prompt == null) return 1;
    const ids: []const u32 = @ptrCast(prompt.?[0..@intCast(n)]);
    for (ids) |t| if (t > std.math.maxInt(i32)) return 1;
    // The reply needs room for one token at least.
    if (ids.len >= e.context) return 1;
    const room: u32 = e.context - @as(u32, @intCast(ids.len));
    const budget: u32 = if (max_tokens <= 0) room else @min(@as(u32, @intCast(max_tokens)), room);
    // What the server asks before it queues a prompt: whether its prefill and reply fit in memory now.
    var bill: abi.Bill = .{};
    if (!admission.table.admit(e.config(), .of(e.entry.id), .of(ids), budget, abi.kv_configured, false, 0, &bill)) return 1;

    const r = gpa.create(Request) catch return 1;
    r.* = .{ .engine = e, .id = id, .cb = callback, .ctx = ctx, .submitter = std.Thread.getCurrentId() };
    {
        e.mu.lockUncancelable(e.io);
        defer e.mu.unlock(e.io);
        if (e.stop or e.findLive(id) != null) {
            gpa.destroy(r);
            return 1;
        }
        e.live.append(gpa, r) catch {
            gpa.destroy(r);
            return 1;
        };
    }
    const greedy = !(temperature > 0);
    const slot = scheduler.submitNow(e.sch, .{
        .model = e.entry,
        .prompt_ids = ids,
        .sink = .{ .ctx = r, .event = onEvent },
        .sampling = .{
            .temperature = if (greedy) 0 else temperature,
            .top_p = if (top_p > 0 and top_p <= 1) top_p else 1,
        },
        .eos_token_ids = e.config().eosTokenSlice(),
        .max_tokens = budget,
    }) catch {
        // Refused: no event of it was passed on, and none follows.
        e.mu.lockUncancelable(e.io);
        defer e.mu.unlock(e.io);
        for (e.live.items, 0..) |x, i| if (x == r) {
            _ = e.live.swapRemove(i);
            break;
        };
        gpa.destroy(r);
        return 1;
    };
    e.mu.lockUncancelable(e.io);
    defer e.mu.unlock(e.io);
    // A request already past its final callback may be freed: only one still live is touched.
    for (e.live.items) |x| if (x == r) {
        r.slot = slot;
        if (r.cancel_wanted) cancelLocked(r);
    };
    return 0;
}

/// Caller holds `mu`, so the request is live and its slot (when the scheduler has one) is not yet handed back. The
/// flag alone: `Slot.cancel` also takes the slot's lock, which the inference thread holds around a final event.
fn cancelLocked(r: *Request) void {
    r.cancel_wanted = true;
    if (r.slot) |s| s.cancelled.store(true, .release);
}

export fn pico_ms_cancel(h: ?*anyopaque, id: u64) void {
    const e: *Engine = @ptrCast(@alignCast(h orelse return));
    e.mu.lockUncancelable(e.io);
    defer e.mu.unlock(e.io);
    if (e.findLive(id)) |r| cancelLocked(r);
}

// ── memory

export fn pico_ms_model_memory(h: ?*anyopaque, out: ?*[3]i64) i32 {
    if (h == null) return -1;
    return lib.pico_ms_memory(out);
}

fn dropPrefixCache(ctx: *anyopaque) void {
    const e: *Engine = @ptrCast(@alignCast(ctx));
    if (e.entry.prefix_cache) |hc| hc.invalidateAll("pico_ms_trim");
    publishHotCacheResidency(e.sch);
}

fn heldBytes() u64 {
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    return active + cache;
}

/// Drops the kept prompt prefixes (on the inference thread, between two passes) and MLX's buffer pool.
export fn pico_ms_trim(h: ?*anyopaque, want_bytes: i64) i64 {
    const e: *Engine = @ptrCast(@alignCast(h orelse return 0));
    if (want_bytes <= 0) return 0;
    const before = heldBytes();
    // The scheduler clears MLX's pool itself once the job has run.
    var job: requests.GenRequest = .{ .ctx = e, .run = dropPrefixCache, .model = e.entry };
    scheduler.runGeneration(e.sch, &job) catch {};
    _ = mlx.mlx_clear_cache();
    return @intCast(before -| heldBytes());
}
