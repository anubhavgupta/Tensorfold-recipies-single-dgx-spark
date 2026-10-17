//! libdory's C ABI as the Dory backend sees it: the symbols, resolved once, and where the library is found.
const std = @import("std");

/// The C ABI version of libdory this backend speaks (`dory_abi`).
pub const abi_version: c_int = 1;

extern "c" fn dlopen(path: [*:0]const u8, mode: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
pub extern "c" fn dlerror() ?[*:0]const u8;
extern "c" fn readlink(path: [*:0]const u8, buf: [*]u8, len: usize) isize;


/// libdory's C ABI, resolved once.
pub const Api = struct {
    open: *const fn ([*:0]const u8, c_int) callconv(.c) ?*anyopaque,
    free: *const fn (?*anyopaque) callconv(.c) void,
    seqs: *const fn (?*anyopaque) callconv(.c) c_int,
    seq_alloc: *const fn (?*anyopaque) callconv(.c) c_int,
    seq_release: *const fn (?*anyopaque, c_int) callconv(.c) void,
    seq_bytes: *const fn (?*anyopaque) callconv(.c) u64,
    prompt_end: *const fn (?*anyopaque, c_int, c_int) callconv(.c) void,
    prefill: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int,
    forward: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, c_int) callconv(.c) c_int,
    forward_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int,
    candidates: *const fn (?*anyopaque, c_int, c_int, [*]f32, [*]c_int) callconv(.c) void,
    keep: *const fn (?*anyopaque, c_int, c_int) callconv(.c) void,
    propose: *const fn (?*anyopaque, c_int, c_int, [*]c_int) callconv(.c) c_int,
    propose_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]c_int, [*]c_int) callconv(.c) void,
    has_draft: *const fn (?*anyopaque) callconv(.c) c_int,
    abi: *const fn () callconv(.c) c_int,
    mask_rows: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const u32, c_int, c_int) callconv(.c) c_int,
    cost: *const fn (?*anyopaque, c_int) callconv(.c) f64,
    draft_multi_cost: *const fn (?*anyopaque) callconv(.c) f64,
    shared_costs: *const fn (?*anyopaque, [*]c_int, [*]f64, c_int) callconv(.c) c_int,
    snap_bytes: *const fn (?*anyopaque, c_int) callconv(.c) u64,
    snap_save: *const fn (?*anyopaque, c_int, c_int) callconv(.c) ?*anyopaque,
    snap_restore: *const fn (?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int,
    snap_drop: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void,

    pub fn load(handle: ?*anyopaque) !Api {
        return .{
            .open = try sym(*const fn ([*:0]const u8, c_int) callconv(.c) ?*anyopaque, handle, "dory_open"),
            .free = try sym(*const fn (?*anyopaque) callconv(.c) void, handle, "dory_free"),
            .seqs = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "dory_seqs"),
            .seq_alloc = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "dory_seq_alloc"),
            .seq_release = try sym(*const fn (?*anyopaque, c_int) callconv(.c) void, handle, "dory_seq_release"),
            .seq_bytes = try sym(*const fn (?*anyopaque) callconv(.c) u64, handle, "dory_seq_bytes"),
            .prompt_end = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) void, handle, "dory_prompt_end"),
            .prefill = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int, handle, "dory_prefill_seq"),
            .forward = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, c_int) callconv(.c) c_int, handle, "dory_forward_seq"),
            .forward_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int, handle, "dory_forward_multi"),
            .candidates = try sym(*const fn (?*anyopaque, c_int, c_int, [*]f32, [*]c_int) callconv(.c) void, handle, "dory_candidates"),
            .keep = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) void, handle, "dory_keep_seq"),
            .propose = try sym(*const fn (?*anyopaque, c_int, c_int, [*]c_int) callconv(.c) c_int, handle, "dory_propose_seq"),
            .propose_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]c_int, [*]c_int) callconv(.c) void, handle, "dory_propose_multi"),
            .has_draft = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "dory_has_draft"),
            .abi = try sym(*const fn () callconv(.c) c_int, handle, "dory_abi"),
            .mask_rows = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const u32, c_int, c_int) callconv(.c) c_int, handle, "dory_mask_rows"),
            .cost = try sym(*const fn (?*anyopaque, c_int) callconv(.c) f64, handle, "dory_cost"),
            .draft_multi_cost = try sym(*const fn (?*anyopaque) callconv(.c) f64, handle, "dory_draft_multi_cost"),
            .shared_costs = try sym(*const fn (?*anyopaque, [*]c_int, [*]f64, c_int) callconv(.c) c_int, handle, "dory_shared_costs"),
            .snap_bytes = try sym(*const fn (?*anyopaque, c_int) callconv(.c) u64, handle, "dory_snap_bytes"),
            .snap_save = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) ?*anyopaque, handle, "dory_snap_save_seq"),
            .snap_restore = try sym(*const fn (?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int, handle, "dory_snap_restore_seq"),
            .snap_drop = try sym(*const fn (?*anyopaque, ?*anyopaque) callconv(.c) void, handle, "dory_snap_drop"),
        };
    }

    fn sym(comptime T: type, handle: ?*anyopaque, name: [*:0]const u8) !T {
        const p = dlsym(handle, name) orelse {
            std.log.err("dory: libdory.so has no {s}", .{name});
            return error.DoryMissing;
        };
        return @ptrCast(@alignCast(p));
    }
};

/// libdory.so: TENSORFOLD_DORY_LIB, else beside the server (native/lib), else the loader's path.
pub fn openLibrary() ?*anyopaque {
    if (std.c.getenv("TENSORFOLD_DORY_LIB")) |p| return dlopen(p, 2);
    var exe: [std.fs.max_path_bytes]u8 = undefined;
    const n = readlink("/proc/self/exe", &exe, exe.len - 1);
    if (n > 0) {
        const dir = std.fs.path.dirname(exe[0..@intCast(n)]) orelse "";
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        if (std.fmt.bufPrintSentinel(&buf, "{s}/../lib/libdory.so", .{dir}, 0)) |path| {
            if (dlopen(path, 2)) |h| return h;
        } else |_| {}
    }
    return dlopen("libdory.so", 2);
}
