//! The MLX text-generation engine behind libPicoMlxServe.dylib (`src/embed_lib.zig`): open a model directory, turn
//! text into token ids and back, and generate from a prompt of token ids. It is the server's own boot
//! (`main.zig`, `host/http/server.zig` `serve`) without the HTTP listener: one scheduler, whose inference thread
//! loads the model and runs every forward, and one registry holding that model.
//!
//! One model is open at a time: the engine's launch state (`server.launch()`, the admission hooks) is one per
//! process, and MLX has one caller, the inference thread.
//!
//! Callbacks run on the inference thread. Inside one, what would wait for that thread is refused at once:
//! `pico_ms_close` does nothing (the handle stays open; close it after the callback returns), `pico_ms_trim` returns
//! -1, `pico_ms_submit` returns 1 (no callback follows), and `pico_ms_open` returns NULL with a message.
//! `pico_ms_cancel`, `pico_ms_tokenize`, `pico_ms_detokenize`, `pico_ms_model_memory`, `pico_ms_can_serve` and
//! `pico_ms_load_bytes` work there as anywhere.
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

/// Set on the inference thread while it runs one of the caller's callbacks: what would wait on that thread is
/// refused there (`pico_ms_close`, `pico_ms_trim`, `pico_ms_submit`, `pico_ms_open`).
threadlocal var in_callback = false;

fn deliver(r: *Request, tokens: ?[*]const i32, n: i32, status: i32, usage: ?*const [6]i64) void {
    in_callback = true;
    defer in_callback = false;
    r.cb(r.ctx, tokens, n, status, usage);
}

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
    /// Past its final callback (under `Engine.mu`).
    finished: bool = false,
    /// Its owners: the submitting call, until it has published the slot, and the engine, until the reaper has handed
    /// the slot back. A request can finish and be reaped before its submitter returns from the scheduler; the last
    /// owner frees it, so the submitter never meets another request at its address.
    refs: std.atomic.Value(u8) = .init(2),
};

const Engine = struct {
    io: std.Io,
    model_dir: []u8,
    registry: *ModelRegistry,
    entry: *LoadedModel,
    sch: *Scheduler,
    /// Prompt and reply together, frozen at load.
    context: u32,
    /// The model's end tokens.
    end_ids: []const u32,
    /// Where requests live.
    alloc: std.mem.Allocator = gpa,
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

    fn newRequest(e: *Engine, id: u64, cb: EventCb, ctx: ?*anyopaque) !*Request {
        const r = try e.alloc.create(Request);
        r.* = .{ .engine = e, .id = id, .cb = cb, .ctx = ctx, .submitter = std.Thread.getCurrentId() };
        return r;
    }

    /// Takes `r` into the live set; false, and nothing changed, when the engine is closing or `r.id` is live.
    fn track(e: *Engine, r: *Request) bool {
        e.mu.lockUncancelable(e.io);
        defer e.mu.unlock(e.io);
        if (e.stop or e.findLive(r.id) != null) return false;
        e.live.append(gpa, r) catch return false;
        return true;
    }

    /// A submission refused after `track`: no event of `r` was passed on and none follows.
    fn untrack(e: *Engine, r: *Request) void {
        e.mu.lockUncancelable(e.io);
        for (e.live.items, 0..) |x, i| if (x == r) {
            _ = e.live.swapRemove(i);
            break;
        };
        e.mu.unlock(e.io);
        e.alloc.destroy(r);
    }

    /// The submitter's last word on `r`: the slot the scheduler gave it, unless `r` has finished (the slot may then
    /// be freed). Lets go of the submitter's reference.
    fn publish(e: *Engine, r: *Request, slot: *Slot) void {
        {
            e.mu.lockUncancelable(e.io);
            defer e.mu.unlock(e.io);
            if (!r.finished) {
                r.slot = slot;
                if (r.cancel_wanted) cancelLocked(r);
            }
        }
        e.release(r);
    }

    fn release(e: *Engine, r: *Request) void {
        if (r.refs.fetchSub(1, .acq_rel) == 1) e.alloc.destroy(r);
    }

    /// `r`'s final callback has returned: it leaves the live set, and the reaper hands its slot back.
    fn retire(e: *Engine, r: *Request, slot: *Slot) void {
        e.mu.lockUncancelable(e.io);
        defer e.mu.unlock(e.io);
        r.slot = slot;
        r.finished = true;
        for (e.live.items, 0..) |x, i| if (x == r) {
            _ = e.live.swapRemove(i);
            break;
        };
        e.done.append(gpa, r) catch @panic("pico_ms: out of memory");
        e.cond.broadcast(e.io);
    }

    /// The reaper's next finished request; null once the engine stops with none left.
    fn takeDone(e: *Engine) ?*Request {
        e.mu.lockUncancelable(e.io);
        defer e.mu.unlock(e.io);
        while (e.done.items.len == 0 and !e.stop) e.cond.waitUncancelable(e.io, &e.mu);
        const r = e.done.pop() orelse return null;
        e.reaping += 1;
        return r;
    }

    /// The reaper has handed `r`'s slot back to the scheduler.
    fn reaped(e: *Engine, r: *Request) void {
        e.release(r);
        e.mu.lockUncancelable(e.io);
        defer e.mu.unlock(e.io);
        e.reaping -= 1;
        e.cond.broadcast(e.io);
    }
};

// ── sink: the scheduler's events for one request, on the inference thread

fn isEnd(e: *const Engine, token: u32) bool {
    for (e.end_ids) |id| if (id == token) return true;
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
                deliver(r, @ptrCast(ids.ptr), @intCast(n), -1, null);
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
            deliver(r, null, 0, @backingInt(status), &usage);
            e.retire(r, slot);
        },
    }
    return .go;
}

/// Hands each finished request's slot back to the scheduler. `complete` waits out the pass that holds the slot, so
/// it cannot run on the inference thread that just finished it.
fn reaperLoop(e: *Engine) void {
    while (e.takeDone()) |r| {
        scheduler.complete(e.sch, r.slot.?);
        e.reaped(r);
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
    if (in_callback) return fail(err, err_len, "not inside a callback: call pico_ms_open after it returns", .{});
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
        .end_ids = entry.config.?.eosTokenSlice(),
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
    // Close waits for every request's final callback, this one's included: here it would wait forever.
    if (in_callback) {
        log.err("pico_ms_close called inside a callback: ignored, the handle stays open\n", .{});
        return;
    }
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
    if (in_callback) return 1;
    if (n <= 0 or prompt == null) return 1;
    const ids: []const u32 = @ptrCast(prompt.?[0..@intCast(n)]);
    for (ids) |t| if (t > std.math.maxInt(i32)) return 1;
    // The reply needs room for one token at least.
    if (ids.len >= e.context) return 1;
    const room: u32 = e.context - @as(u32, @intCast(ids.len));
    const budget: u32 = if (max_tokens <= 0) room else @min(@as(u32, @intCast(max_tokens)), room);

    const r = e.newRequest(id, callback, ctx) catch return 1;
    if (!e.track(r)) {
        e.alloc.destroy(r);
        return 1;
    }
    // What the server asks before it queues a prompt: whether its prefill and reply fit in memory now.
    var bill: abi.Bill = .{};
    if (!admission.table.admit(e.config(), .of(e.entry.id), .of(ids), budget, abi.kv_configured, false, 0, &bill)) {
        e.untrack(r);
        return 1;
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
        // Refused: no event of it was passed on (the refused slot's own came on this thread), and none follows.
        e.untrack(r);
        return 1;
    };
    e.publish(r, slot);
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

/// Drops the kept prompt prefixes (on the inference thread, between two passes) and MLX's buffer pool. -1 inside a
/// callback.
export fn pico_ms_trim(h: ?*anyopaque, want_bytes: i64) i64 {
    const e: *Engine = @ptrCast(@alignCast(h orelse return 0));
    // The trim runs on the inference thread, which is running this callback.
    if (in_callback) return -1;
    if (want_bytes <= 0) return 0;
    const before = heldBytes();
    // The scheduler clears MLX's pool itself once the job has run.
    var job: requests.GenRequest = .{ .ctx = e, .run = dropPrefixCache, .model = e.entry };
    scheduler.runGeneration(e.sch, &job) catch {};
    _ = mlx.mlx_clear_cache();
    return @intCast(before -| heldBytes());
}

// ── tests (`zig build pico-test`; `PICO_MS_TEST_MODEL=<dir>` adds the ones that need a model)

const testing = std.testing;
const env = @import("env.zig");

fn ignoreEvent(_: ?*anyopaque, _: ?[*]const i32, _: i32, _: i32, _: ?*const [6]i64) callconv(.c) void {}

test "pico_ms: calls without a model or a handle refuse, and say so" {
    try testing.expectEqual(@as(i32, 2), lib.pico_ms_abi_version());
    try testing.expectEqual(@as(i32, 0), pico_ms_can_serve(null));
    try testing.expectEqual(@as(i32, 0), pico_ms_can_serve("relative/dir"));
    try testing.expectEqual(@as(i32, 0), pico_ms_can_serve("/"));
    try testing.expectEqual(@as(i64, -1), pico_ms_load_bytes("/"));
    var err: [128]u8 = @splat(0xaa);
    try testing.expect(pico_ms_open("relative/dir", &err, err.len) == null);
    try testing.expectEqualStrings("the model directory must be an absolute path", std.mem.sliceTo(&err, 0));
    try testing.expect(pico_ms_open("/", &err, err.len) == null);
    try testing.expect(std.mem.sliceTo(&err, 0).len > 0);
    // A message that does not fit is cut, and still ends in NUL.
    var small: [4]u8 = @splat(0xaa);
    try testing.expect(pico_ms_open("relative", &small, small.len) == null);
    try testing.expectEqualStrings("the", std.mem.sliceTo(&small, 0));
    pico_ms_close(null);
    pico_ms_cancel(null, 1);
    var ids: [4]i32 = undefined;
    try testing.expectEqual(@as(i32, -1), pico_ms_tokenize(null, "a", 1, &ids, ids.len));
    try testing.expectEqual(@as(i32, -1), pico_ms_detokenize(null, &ids, 1, null, 0));
    try testing.expectEqual(@as(i32, 1), pico_ms_submit(null, 1, &ids, 1, 1, 0, 0, ignoreEvent, null));
    var mem: [3]i64 = undefined;
    try testing.expectEqual(@as(i32, -1), pico_ms_model_memory(null, &mem));
    try testing.expectEqual(@as(i64, 0), pico_ms_trim(null, 1));
}

/// An engine with no model behind it: no scheduler, no reaper, for the request bookkeeping alone.
fn ledgerEngine(alloc: std.mem.Allocator) Engine {
    return .{ .io = theIo(), .model_dir = &.{}, .registry = undefined, .entry = undefined, .sch = undefined, .context = 0, .end_ids = &.{}, .alloc = alloc };
}

/// Request 1 finishes on the inference thread and the reaper hands its slot back, all before the thread that submitted
/// it gets back from the scheduler; request 2 is submitted meanwhile; then request 1's submitter publishes its slot.
fn finishBeforePublish(alloc: std.mem.Allocator) !void {
    var e = ledgerEngine(alloc);
    defer e.live.deinit(gpa);
    defer e.done.deinit(gpa);
    var slot1: Slot = undefined;
    slot1.cancelled = .init(false);

    const r1 = try e.newRequest(1, ignoreEvent, null);
    try testing.expect(e.track(r1));
    e.retire(r1, &slot1);
    e.reaped(e.takeDone().?);
    const r2 = try e.newRequest(2, ignoreEvent, null);
    try testing.expect(e.track(r2));
    e.publish(r1, &slot1);

    try testing.expect(r2.slot == null);
    pico_ms_cancel(&e, 2);
    try testing.expect(!slot1.cancelled.load(.acquire));
    try testing.expect(r2.cancel_wanted);
    e.untrack(r2);
    try testing.expectEqual(@as(usize, 0), e.live.items.len);
}

test "pico_ms: a request that finishes before its submitter publishes is never taken for the next one" {
    // An allocator that hands a freed request's memory to the next request, as malloc may.
    var buf: [4096]u8 align(64) = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&buf);
    try finishBeforePublish(fba.allocator());
    // And each request is freed exactly once (the test allocator reports a leak or a double free).
    try finishBeforePublish(testing.allocator);
}

test "pico_ms: a live id is refused, and a closing engine takes no request" {
    var e = ledgerEngine(gpa);
    defer e.live.deinit(gpa);
    const r1 = try e.newRequest(7, ignoreEvent, null);
    try testing.expect(e.track(r1));
    const dup = try e.newRequest(7, ignoreEvent, null);
    try testing.expect(!e.track(dup));
    e.alloc.destroy(dup);
    e.untrack(r1);
    // Its id is free again once it has left.
    const again = try e.newRequest(7, ignoreEvent, null);
    try testing.expect(e.track(again));
    e.untrack(again);
    e.stop = true;
    const late = try e.newRequest(8, ignoreEvent, null);
    try testing.expect(!e.track(late));
    e.alloc.destroy(late);
}

/// An engine whose scheduler runs with no model, as a headless server boots: its inference thread idles and runs
/// jobs, and the reaper runs. Nothing it holds needs the GPU.
const Bare = struct {
    e: *Engine,
    stub: gen_mod.StubCpuState,

    fn open() !Bare {
        lib.ensureInit();
        const io = theIo();
        var stub = try gen_mod.buildStubCpuState(gpa, .image);
        errdefer gen_mod.freeStubCpuState(gpa, &stub);
        const registry = try ModelRegistry.init(gpa, io, null, 1, 0, null);
        errdefer registry.deinit();
        const entry = try registry.registerStub("bare", "/nonexistent/pico-ms-bare", null);
        const sch = try scheduler.init(gpa, io, .{
            .registry = registry,
            .entry = entry,
            .config = stub.config,
            .tok = stub.tok,
            .chat_config = stub.chat_config,
            .model_dir = "",
            .no_initial_load = true,
            .warmup_eager = false,
            .launch_warmup_eager = false,
        }, 1);
        errdefer scheduler.deinit(sch);
        const e = try gpa.create(Engine);
        e.* = .{ .io = io, .model_dir = &.{}, .registry = registry, .entry = entry, .sch = sch, .context = 4096, .end_ids = &.{} };
        e.reaper = try std.Thread.spawn(.{}, reaperLoop, .{e});
        return .{ .e = e, .stub = stub };
    }

    fn close(b: *Bare) void {
        const e = b.e;
        e.mu.lockUncancelable(e.io);
        e.stop = true;
        e.cond.broadcast(e.io);
        e.mu.unlock(e.io);
        e.reaper.?.join();
        scheduler.deinit(e.sch);
        e.registry.deinit();
        gen_mod.freeStubCpuState(gpa, &b.stub);
        e.live.deinit(gpa);
        e.done.deinit(gpa);
        gpa.destroy(e);
    }
};

/// Runs `run(ctx)` on `e`'s inference thread, where callbacks run, and waits for it at most 10 s.
fn onInferenceThread(e: *Engine, run: *const fn (*anyopaque) void, ctx: *anyopaque) !void {
    const Job = struct {
        job: requests.GenRequest,
        sch: *Scheduler,
        finished: std.atomic.Value(bool) = .init(false),
        fn post(j: *@This()) void {
            scheduler.runGeneration(j.sch, &j.job) catch {};
            j.finished.store(true, .release);
        }
    };
    // On a timeout the stuck threads still hold it: it is left to them.
    const j = try gpa.create(Job);
    j.* = .{ .job = .{ .ctx = ctx, .run = run, .model = e.entry }, .sch = e.sch };
    const t = try std.Thread.spawn(.{}, Job.post, .{j});
    var waited_ms: u32 = 0;
    while (!j.finished.load(.acquire)) : (waited_ms += 10) {
        if (waited_ms >= 10_000) {
            t.detach();
            return error.Deadlocked;
        }
        std.Io.sleep(theIo(), .fromMilliseconds(10), .awake) catch {};
    }
    t.join();
    gpa.destroy(j);
}

/// A caller's callback that calls back into the library.
const Reentry = struct {
    e: *Engine,
    call: enum { close, trim, submit_open },
    trimmed: i64 = 0,
    submitted: i32 = 0,
    opened: ?*anyopaque = null,
    err: [128]u8 = @splat(0),
    returned: bool = false,

    fn cb(ctx: ?*anyopaque, _: ?[*]const i32, _: i32, _: i32, _: ?*const [6]i64) callconv(.c) void {
        const p: *Reentry = @ptrCast(@alignCast(ctx.?));
        switch (p.call) {
            .close => pico_ms_close(p.e),
            .trim => p.trimmed = pico_ms_trim(p.e, 1 << 30),
            .submit_open => {
                const prompt = [_]i32{ 1, 2, 3 };
                p.submitted = pico_ms_submit(p.e, 99, &prompt, prompt.len, 4, 0, 0, ignoreEvent, null);
                p.opened = pico_ms_open("/nonexistent/pico-ms", &p.err, p.err.len);
            },
        }
        p.returned = true;
    }
};

/// One batch for the request at `ctx`, delivered as the scheduler delivers it.
fn deliverBatch(ctx: *anyopaque) void {
    const one = [_]u32{42};
    const ev: abi.Event = .{ .kind = .tokens, .tokens = abi.span(u32, &one) };
    _ = onEvent(ctx, 0, &ev);
}

fn reenter(call: @FieldType(Reentry, "call")) !Reentry {
    var bare = try Bare.open();
    const e = bare.e;
    var p: Reentry = .{ .e = e, .call = call };
    const r = try e.newRequest(1, Reentry.cb, &p);
    try testing.expect(e.track(r));
    // A deadlock leaves the engine to its stuck threads.
    try onInferenceThread(e, deliverBatch, r);
    try testing.expect(p.returned);
    // The engine is still open and still has its request.
    try testing.expect(e.findLive(1) == r);
    e.untrack(r);
    bare.close();
    return p;
}

test "pico_ms: closing the handle from inside a callback returns at once and leaves it open" {
    _ = try reenter(.close);
}

test "pico_ms: trimming from inside a callback is refused at once" {
    const p = try reenter(.trim);
    try testing.expectEqual(@as(i64, -1), p.trimmed);
}

test "pico_ms: submitting or opening from inside a callback is refused" {
    const p = try reenter(.submit_open);
    try testing.expectEqual(@as(i32, 1), p.submitted);
    try testing.expect(p.opened == null);
    try testing.expectEqualStrings("not inside a callback: call pico_ms_open after it returns", std.mem.sliceTo(&p.err, 0));
}

// ── with a model (`PICO_MS_TEST_MODEL`)

/// What one request delivered, filled from the inference thread.
const Seen = struct {
    tokens: std.atomic.Value(i64) = .init(0),
    batches: std.atomic.Value(u32) = .init(0),
    finals: std.atomic.Value(u32) = .init(0),
    late: std.atomic.Value(u32) = .init(0),
    status: std.atomic.Value(i32) = .init(-99),
    usage: [6]i64 = @splat(0),

    fn cb(ctx: ?*anyopaque, tokens: ?[*]const i32, n: i32, status: i32, usage: ?*const [6]i64) callconv(.c) void {
        const s: *Seen = @ptrCast(@alignCast(ctx.?));
        if (s.finals.load(.acquire) > 0) _ = s.late.fetchAdd(1, .acq_rel);
        if (status == -1) {
            std.debug.assert(tokens != null and n > 0 and usage == null);
            _ = s.tokens.fetchAdd(n, .acq_rel);
            _ = s.batches.fetchAdd(1, .acq_rel);
            return;
        }
        s.usage = usage.?.*;
        s.status.store(status, .release);
        _ = s.finals.fetchAdd(1, .acq_rel);
    }

    fn wait(s: *Seen, what: enum { batch, final }) !void {
        var ms: u32 = 0;
        while (ms < 120_000) : (ms += 2) {
            switch (what) {
                .batch => if (s.batches.load(.acquire) > 0 or s.finals.load(.acquire) > 0) return,
                .final => if (s.finals.load(.acquire) > 0) return,
            }
            std.Io.sleep(theIo(), .fromMilliseconds(2), .awake) catch {};
        }
        return error.Timeout;
    }
};

fn openTestModel() !?*anyopaque {
    const dir = env.get(.PICO_MS_TEST_MODEL) orelse return null;
    var err: [256]u8 = @splat(0);
    const h = pico_ms_open(dir, &err, err.len) orelse {
        std.debug.print("pico_ms_open({s}): {s}\n", .{ dir, std.mem.sliceTo(&err, 0) });
        return error.OpenFailed;
    };
    return h;
}

fn promptOf(h: *anyopaque, text: []const u8, out: []i32) []const i32 {
    const n = pico_ms_tokenize(h, text.ptr, @intCast(text.len), out.ptr, @intCast(out.len));
    return out[0..@intCast(n)];
}

test "pico_ms: with a model: refusals get no callback; generation, a live id, cancel and close behave" {
    const h = try openTestModel() orelse return error.SkipZigTest;
    const e: *Engine = @ptrCast(@alignCast(h));
    var buf: [64]i32 = undefined;
    const prompt = promptOf(h, "The capital of France is", &buf);

    // Refused: a prompt that leaves the reply no room, an empty one. No callback, ever.
    var refused: Seen = .{};
    const long = try gpa.alloc(i32, e.context);
    defer gpa.free(long);
    @memset(long, prompt[0]);
    try testing.expectEqual(@as(i32, 1), pico_ms_submit(h, 1, long.ptr, @intCast(long.len), 8, 0, 0, Seen.cb, &refused));
    try testing.expectEqual(@as(i32, 1), pico_ms_submit(h, 1, prompt.ptr, 0, 8, 0, 0, Seen.cb, &refused));

    // Runs to max_tokens.
    var a: Seen = .{};
    try testing.expectEqual(@as(i32, 0), pico_ms_submit(h, 1, prompt.ptr, @intCast(prompt.len), 8, 0, 0, Seen.cb, &a));
    try a.wait(.final);
    try testing.expectEqual(@as(i32, 3), a.status.load(.acquire));
    try testing.expectEqual(@as(i64, 8), a.tokens.load(.acquire));
    try testing.expectEqual(@as(i64, 8), a.usage[2]);
    try testing.expectEqual(@as(i64, @intCast(prompt.len)), a.usage[0] + a.usage[1]);

    // A live id is refused; once cancelled, its final callback still comes, and the id is free again.
    var b: Seen = .{};
    var dup: Seen = .{};
    try testing.expectEqual(@as(i32, 0), pico_ms_submit(h, 2, prompt.ptr, @intCast(prompt.len), 0, 0, 0, Seen.cb, &b));
    try b.wait(.batch);
    try testing.expectEqual(@as(i32, 1), pico_ms_submit(h, 2, prompt.ptr, @intCast(prompt.len), 8, 0, 0, Seen.cb, &dup));
    pico_ms_cancel(h, 2);
    try b.wait(.final);
    try testing.expectEqual(@as(i32, 1), b.status.load(.acquire));
    try testing.expectEqual(b.tokens.load(.acquire), b.usage[2]);

    // Close with a request running: it is cancelled and has its final callback before close returns.
    var c: Seen = .{};
    try testing.expectEqual(@as(i32, 0), pico_ms_submit(h, 2, prompt.ptr, @intCast(prompt.len), 0, 0, 0, Seen.cb, &c));
    try c.wait(.batch);
    pico_ms_close(h);
    try testing.expectEqual(@as(u32, 1), c.finals.load(.acquire));
    try testing.expectEqual(@as(i32, 1), c.status.load(.acquire));

    for ([_]*Seen{ &refused, &dup }) |s| try testing.expectEqual(@as(u32, 0), s.finals.load(.acquire) + s.batches.load(.acquire));
    for ([_]*Seen{ &a, &b, &c }) |s| try testing.expectEqual(@as(u32, 0), s.late.load(.acquire));
}

test "pico_ms: with a model: racing submitters each get exactly one final callback per accepted request" {
    const h = try openTestModel() orelse return error.SkipZigTest;
    defer pico_ms_close(h);
    var buf: [64]i32 = undefined;
    const prompt = promptOf(h, "The capital of France is", &buf);
    const per_thread = 32;
    const Submitter = struct {
        seen: [per_thread]Seen = @splat(.{}),
        accepted: [per_thread]bool = @splat(false),

        fn run(s: *@This(), handle: *anyopaque, base: u64, ids: []const i32) void {
            for (&s.seen, &s.accepted, 0..) |*seen, *ok, i| {
                const id = base + i;
                // Short requests: they finish while their submitter is still on its way out of submit.
                ok.* = pico_ms_submit(handle, id, ids.ptr, @intCast(ids.len), 1 + @as(i32, @intCast(i % 3)), 0, 0, Seen.cb, seen) == 0;
                if (i % 4 == 0) pico_ms_cancel(handle, id);
            }
        }
    };
    var subs: [4]Submitter = @splat(.{});
    var threads: [4]std.Thread = undefined;
    for (&subs, &threads, 0..) |*s, *t, k| t.* = try std.Thread.spawn(.{}, Submitter.run, .{ s, h, @as(u64, k) * 1000, prompt });
    for (threads) |t| t.join();
    var accepted: u32 = 0;
    for (&subs) |*s| for (&s.seen, s.accepted) |*seen, ok| {
        if (!ok) {
            try testing.expectEqual(@as(u32, 0), seen.finals.load(.acquire) + seen.batches.load(.acquire));
            continue;
        }
        accepted += 1;
        try seen.wait(.final);
        try testing.expectEqual(@as(u32, 1), seen.finals.load(.acquire));
        try testing.expectEqual(@as(u32, 0), seen.late.load(.acquire));
        try testing.expectEqual(seen.tokens.load(.acquire), seen.usage[2]);
    };
    try testing.expect(accepted > 0);
}
