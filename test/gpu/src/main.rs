// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: BSD-3-Clause

//! The GPU conformance run: every case of the corpus `Cases.hs` writes,
//! compiled with the options its lowering asks for (or, with
//! `--default-options`, Metal's defaults), launched on this Mac's GPU as
//! `FuryKernel.Metal`'s documentation tells a runtime to, and its outputs
//! compared with the interpreter's by the leave the IR gives a target. Run
//! as `nix run .#gpu-conformance`: plain, under `MTL_SHADER_VALIDATION=1`
//! and with Metal's default options.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::{env, fs, process};

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{
    NSError, NSFastEnumeration, NSFastEnumerationState, NSProcessInfo, NSString,
};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLResourceUsage, MTLShaderValidation, MTLSize,
};

/// An element type of a buffer or a table's arrays.
#[derive(Clone, Copy, PartialEq)]
enum Ty {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    I8,
    U8,
    Bool,
}

impl Ty {
    fn parse(s: &str) -> Ty {
        match s {
            "f32" => Ty::F32,
            "f16" => Ty::F16,
            "bf16" => Ty::Bf16,
            "i32" => Ty::I32,
            "u32" => Ty::U32,
            "i8" => Ty::I8,
            "u8" => Ty::U8,
            "bool" => Ty::Bool,
            _ => fail(&format!("an element type {s}")),
        }
    }

    fn size(self) -> usize {
        match self {
            Ty::F32 | Ty::I32 | Ty::U32 => 4,
            Ty::F16 | Ty::Bf16 => 2,
            Ty::I8 | Ty::U8 | Ty::Bool => 1,
        }
    }
}

/// What of a NaN the IR leaves a target.
#[derive(Default)]
enum Nan {
    /// Nothing: bit for bit.
    #[default]
    Bits,
    /// A NaN out of arithmetic: its sign and payload are the target's, and
    /// it is quiet or signalling as the interpreter's is (the core's class).
    Class,
    /// All of it, its quiet bit included: a conversion's, a minimum's or a
    /// maximum's.
    Any,
}

/// How an output is held to the interpreter's: the IR's leave to a target.
#[derive(Default)]
struct Rule {
    nan: Nan,
    /// binary32 (and bfloat16) subnormals may be zeros of their signs: in
    /// operands (the interpreter's run on flushed inputs, `alt`, which is
    /// taken only here) and in results.
    ftz: bool,
    /// An operand returned (select, minima, maxima): under `ftz`, a
    /// subnormal as it was where the flushed run gives the zero of its sign
    /// (of either sign, with `zeros`).
    operand: bool,
    /// Of -0 and +0, either may come out.
    zeros: bool,
    /// Within this many units in the last place of the interpreter's.
    ulps: u32,
    /// Within a bound of its own for each element (f64, `bound`).
    bound: Option<PathBuf>,
}

impl Rule {
    fn parse(s: &str, dir: &Path) -> Rule {
        let mut r = Rule::default();
        for part in s.split('+') {
            match part {
                "bits" => {}
                "class" => r.nan = Nan::Class,
                "nan" => r.nan = Nan::Any,
                "ftz" => r.ftz = true,
                "operand" => r.operand = true,
                "zeros" => r.zeros = true,
                p if p.starts_with("ulp") => r.ulps = p[3..].parse().unwrap_or_else(|_| fail(p)),
                p if p.starts_with("bound:") => r.bound = Some(dir.join(&p[6..])),
                p => fail(&format!("a rule {p}")),
            }
        }
        r
    }
}

/// A buffer binding, or a table entry's array.
struct Array {
    ty: Ty,
    bytes: usize,
    input: Option<PathBuf>,
    output: Option<(PathBuf, Option<PathBuf>, Rule)>,
}

struct Table {
    index: usize,
    write: bool,
    entries: usize,
    arrays: BTreeMap<usize, Array>,
}

struct Case {
    name: String,
    dir: PathBuf,
    entry: String,
    threads: usize,
    groups: usize,
    tg_memory: usize,
    simd: usize,
    params: Option<(usize, PathBuf)>,
    buffers: Vec<(usize, String, Array)>,
    tables: Vec<(String, Table)>,
}

fn fail(what: &str) -> ! {
    eprintln!("gpu-conformance: {what}");
    process::exit(2)
}

fn read(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| fail(&format!("{}: {e}", p.display())))
}

/// A file of the bytes its array takes, or of a bound for each element.
fn read_exactly(p: &Path, bytes: usize) -> Vec<u8> {
    let b = read(p);
    if b.len() != bytes {
        fail(&format!("{}: {} bytes, not {bytes}", p.display(), b.len()));
    }
    b
}

fn parse_case(dir: &Path) -> Case {
    let text = String::from_utf8(read(&dir.join("case.txt"))).unwrap_or_else(|_| fail("case.txt"));
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_owned();
    let mut case = Case {
        name,
        dir: dir.to_owned(),
        entry: String::new(),
        threads: 0,
        groups: 0,
        tg_memory: 0,
        simd: 0,
        params: None,
        buffers: Vec::new(),
        tables: Vec::new(),
    };
    let file = |s: &str| if s == "-" { None } else { Some(dir.join(s)) };
    let num = |s: &str| {
        s.parse::<usize>()
            .unwrap_or_else(|_| fail(&format!("a number {s}")))
    };
    // The fields a buffer's line and an array's share.
    let array = |fields: &[&str]| match fields {
        [ty, bytes, input, output, alt, rule] => Array {
            ty: Ty::parse(ty),
            bytes: num(bytes),
            input: file(input),
            output: file(output).map(|o| (o, file(alt), Rule::parse(rule, dir))),
        },
        _ => fail(&format!("an array {}", fields.join(" "))),
    };
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["entry", e] => (*e).clone_into(&mut case.entry),
            ["threads", n] => case.threads = num(n),
            ["groups", n] => case.groups = num(n),
            ["tg_memory", n] => case.tg_memory = num(n),
            ["simd", n] => case.simd = num(n),
            ["params", i, f] => case.params = Some((num(i), dir.join(f))),
            ["buffer", i, n, fields @ ..] => {
                case.buffers.push((num(i), (*n).to_owned(), array(fields)));
            }
            ["table", i, n, access, entries] => case.tables.push((
                (*n).to_owned(),
                Table {
                    index: num(i),
                    write: *access != "read",
                    entries: num(entries),
                    arrays: BTreeMap::new(),
                },
            )),
            ["array", n, e, fields @ ..] => {
                let a = array(fields);
                match case.tables.iter_mut().find(|(t, _)| t == n) {
                    Some((_, table)) => {
                        table.arrays.insert(num(e), a);
                    }
                    None => fail(&format!("{}: an array of no table {n}", case.name)),
                }
            }
            [] => {}
            _ => fail(&format!("{}: a line {line}", case.name)),
        }
    }
    case
}

/// The bits of element `i` of a buffer of `ty`, widened to a u64.
fn bits(b: &[u8], ty: Ty, i: usize) -> u64 {
    let s = ty.size();
    let mut v = 0u64;
    for (k, byte) in b[i * s..(i + 1) * s].iter().enumerate() {
        v |= u64::from(*byte) << (8 * k);
    }
    v
}

fn is_nan(ty: Ty, v: u64) -> bool {
    match ty {
        Ty::F32 => (v & 0x7f80_0000) == 0x7f80_0000 && (v & 0x007f_ffff) != 0,
        Ty::F16 => (v & 0x7c00) == 0x7c00 && (v & 0x03ff) != 0,
        Ty::Bf16 => (v & 0x7f80) == 0x7f80 && (v & 0x007f) != 0,
        _ => false,
    }
}

/// Of a NaN, whether it is quiet: the mantissa's top bit.
fn is_quiet(ty: Ty, v: u64) -> bool {
    match ty {
        Ty::F32 => v & 0x0040_0000 != 0,
        Ty::F16 => v & 0x0200 != 0,
        Ty::Bf16 => v & 0x0040 != 0,
        _ => false,
    }
}

fn is_inf(ty: Ty, v: u64) -> bool {
    match ty {
        Ty::F32 => v & 0x7fff_ffff == 0x7f80_0000,
        Ty::F16 => v & 0x7fff == 0x7c00,
        Ty::Bf16 => v & 0x7fff == 0x7f80,
        _ => false,
    }
}

fn is_zero(ty: Ty, v: u64) -> bool {
    match ty {
        Ty::F32 => matches!(v, 0 | 0x8000_0000),
        Ty::F16 | Ty::Bf16 => matches!(v, 0 | 0x8000),
        _ => false,
    }
}

/// A binary32 or bfloat16 subnormal: the subnormals the IR lets a target flush.
fn is_subnormal(ty: Ty, v: u64) -> bool {
    match ty {
        Ty::F32 => (v & 0x7f80_0000) == 0 && (v & 0x007f_ffff) != 0,
        Ty::Bf16 => (v & 0x7f80) == 0 && (v & 0x007f) != 0,
        _ => false,
    }
}

/// ±2^-126, the least binary32 or bfloat16 normal.
fn is_least_normal(ty: Ty, v: u64) -> bool {
    match ty {
        Ty::F32 => v & 0x7fff_ffff == 0x0080_0000,
        Ty::Bf16 => v & 0x7fff == 0x0080,
        _ => false,
    }
}

/// The zero of a value's sign.
fn signed_zero(ty: Ty, v: u64) -> u64 {
    match ty {
        Ty::F32 => v & 0x8000_0000,
        _ => v & 0x8000,
    }
}

fn to_f64(ty: Ty, v: u64) -> f64 {
    match ty {
        Ty::F32 => f64::from(f32::from_bits(u32::try_from(v).unwrap_or(0))),
        Ty::Bf16 => f64::from(f32::from_bits(u32::try_from(v << 16).unwrap_or(0))),
        Ty::F16 => {
            let h = u16::try_from(v).unwrap_or(0);
            let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
            let e = i32::from((h >> 10) & 0x1f);
            let m = f64::from(h & 0x3ff);
            match e {
                0 => sign * m * 2f64.powi(-24),
                31 => {
                    if m == 0.0 {
                        sign * f64::INFINITY
                    } else {
                        f64::NAN
                    }
                }
                _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
            }
        }
        Ty::I32 => f64::from(u32::try_from(v).unwrap_or(0).cast_signed()),
        Ty::I8 => f64::from(u8::try_from(v).unwrap_or(0).cast_signed()),
        Ty::U32 | Ty::U8 | Ty::Bool => f64::from(u32::try_from(v).unwrap_or(0)),
    }
}

/// The distance between two finite values of a float type in units in the
/// last place: of `want`'s binade, or of the one below where `want` is a
/// power of two, since a result just below it, rounded up to it, has the
/// lower binade's unit, half `want`'s.
fn ulp_distance(ty: Ty, got: u64, want: u64) -> f64 {
    // The exponent field's place and mask, its bias and the mantissa's bits.
    let (place, mask, bias, mantissa) = match ty {
        Ty::F32 => (23, 0xff, 127, 23),
        Ty::F16 => (10, 0x1f, 15, 10),
        Ty::Bf16 => (7, 0xff, 127, 7),
        _ => return f64::INFINITY,
    };
    let field = (want >> place) & mask;
    // A subnormal's (or a zero's) unit is the least normal's, as is the
    // unit below the least normal.
    let mut exponent = i32::try_from(field).unwrap_or(0).max(1) - bias;
    if field > 1 && want & ((1 << mantissa) - 1) == 0 {
        exponent -= 1;
    }
    (to_f64(ty, got) - to_f64(ty, want)).abs() / 2f64.powi(exponent - mantissa)
}

/// An element of an output against the interpreter's (`want`) and its run
/// on flushed inputs (`alt`), under the output's rule.
fn agrees(ty: Ty, rule: &Rule, got: u64, want: u64, alt: Option<u64>, bound: Option<f64>) -> bool {
    // The flushed run, only where the rule lets subnormals be flushed.
    let alt = alt.filter(|_| rule.ftz);
    std::iter::once(want).chain(alt).any(|w| {
        if got == w {
            return true;
        }
        if is_nan(ty, got) && is_nan(ty, w) {
            return match rule.nan {
                Nan::Bits => false,
                Nan::Class => is_quiet(ty, got) == is_quiet(ty, w),
                Nan::Any => true,
            };
        }
        // A subnormal result as the zero of its sign, and, tininess being
        // detected before rounding, the least normal too: a result whose
        // exact magnitude is below 2^-126 may come out as a zero where the
        // interpreter rounds it to ±2^-126. The runner sees only the rounded
        // value, so an exact ±2^-126 given as a zero passes as well, a
        // looseness of this rule.
        if rule.ftz && (is_subnormal(ty, w) || is_least_normal(ty, w)) && got == signed_zero(ty, w)
        {
            return true;
        }
        if rule.zeros && is_zero(ty, got) && is_zero(ty, w) {
            return true;
        }
        // An operand returned unflushed where the flushed operands gave its zero.
        if rule.operand
            && Some(w) == alt
            && is_subnormal(ty, got)
            && is_zero(ty, w)
            && (rule.zeros || signed_zero(ty, got) == w)
        {
            return true;
        }
        // A NaN or an infinity only as itself or by the rules above, however
        // far a bound (infinite where the products' magnitudes are) reaches.
        if is_nan(ty, got) || is_nan(ty, w) || is_inf(ty, got) || is_inf(ty, w) {
            return false;
        }
        if rule.ulps > 0 && ulp_distance(ty, got, w) <= f64::from(rule.ulps) {
            return true;
        }
        if let Some(b) = bound {
            return (to_f64(ty, got) - to_f64(ty, w)).abs() <= b;
        }
        false
    })
}

struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Compiles with Metal's default options, not the lowered ones.
    default_options: bool,
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// A launch's buffers: one per binding, in the case's order, and per table
/// its addresses and the arrays they point at.
struct Memory {
    buffers: Vec<Buffer>,
    tables: Vec<(Buffer, BTreeMap<usize, Buffer>)>,
}

fn buffer(gpu: &Gpu, bytes: usize, contents: Option<&[u8]>) -> Buffer {
    // Past the data, and where an array has no initial contents, 0xff
    // bytes: a NaN in every float type and -1 in every integer one. The
    // corpus poisons outputs itself.
    let len = bytes.max(16);
    let b = gpu
        .device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .unwrap_or_else(|| fail("a buffer"));
    let p: NonNull<c_void> = b.contents();
    // SAFETY: the buffer is shared, `len` bytes long and not in use by the GPU yet.
    let dst = unsafe { std::slice::from_raw_parts_mut(p.as_ptr().cast::<u8>(), len) };
    dst.fill(0xff);
    if let Some(c) = contents {
        dst[..c.len()].copy_from_slice(c);
    }
    b
}

fn contents(b: &ProtocolObject<dyn MTLBuffer>, bytes: usize) -> Vec<u8> {
    let p: NonNull<c_void> = b.contents();
    // SAFETY: the command buffer has completed, so the GPU no longer writes it.
    unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), bytes) }.to_vec()
}

/// Compiles a case's source with the lowered options, MSL 4.1 and safe
/// math (fastMathEnabled = false), or with Metal's defaults but for the
/// language version, fast math included, as a runtime that ignores the
/// lowered options does. Returns the library and the compiler's warnings,
/// which fail the case.
fn compile(
    gpu: &Gpu,
    source: &str,
) -> Result<(Retained<ProtocolObject<dyn MTLLibrary>>, String), String> {
    let opts = MTLCompileOptions::new();
    // MSL 4.1, macOS 27's: objc2-metal 0.3.2 names versions up to 4.0 only,
    // and the value is (major << 16) + minor.
    opts.setLanguageVersion(MTLLanguageVersion((4 << 16) + 1));
    if !gpu.default_options {
        opts.setMathMode(MTLMathMode::Safe);
        #[allow(deprecated)]
        opts.setFastMathEnabled(false);
    }
    let src = NSString::from_str(source);
    let mut err: *mut NSError = std::ptr::null_mut();
    // SAFETY: newLibraryWithSource:options:error: takes an NSString, the
    // options and an error out-pointer, which it may set on success too,
    // to the compiler's warnings.
    let lib: Option<Retained<ProtocolObject<dyn MTLLibrary>>> = unsafe {
        msg_send![&*gpu.device, newLibraryWithSource: &*src, options: &*opts, error: &mut err]
    };
    // SAFETY: a non-null error is an autoreleased NSError the call returned.
    let log = unsafe { err.as_ref() }
        .map(|e| e.localizedDescription().to_string())
        .unwrap_or_default();
    match lib {
        Some(l) => Ok((l, log)),
        None => Err(log),
    }
}

/// The case's pipeline, compiled and held to the checks a runtime makes
/// (FuryKernel.Metal), and whether Metal validates its shaders.
fn prepare(
    gpu: &Gpu,
    case: &Case,
) -> Result<(Retained<ProtocolObject<dyn MTLComputePipelineState>>, bool), String> {
    let source = String::from_utf8(read(&case.dir.join("source.metal")))
        .map_err(|_| "source.metal".to_owned())?;
    let (lib, warnings) =
        compile(gpu, &source).map_err(|e| format!("refused by the compiler: {e}"))?;
    if !warnings.is_empty() {
        return Err(format!("compiler warnings: {warnings}"));
    }
    let function = lib
        .newFunctionWithName(&NSString::from_str(&case.entry))
        .ok_or("no entry point")?;
    let pipeline = gpu
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("no pipeline: {}", e.localizedDescription()))?;
    if pipeline.maxTotalThreadsPerThreadgroup() < case.threads {
        return Err(format!(
            "{} threads, the pipeline takes {}",
            case.threads,
            pipeline.maxTotalThreadsPerThreadgroup()
        ));
    }
    if pipeline.threadExecutionWidth() != case.simd {
        return Err(format!(
            "SIMD width {}, the kernel's {}",
            pipeline.threadExecutionWidth(),
            case.simd
        ));
    }
    // The launch table's threadgroup memory is the most a pipeline takes,
    // not what it takes: Metal drops the arrays a kernel doesn't use. Shader
    // validation about doubles a pipeline's, so it is checked only where
    // Metal says it validates nothing, however the environment asked for it.
    // SAFETY: the pipeline's property is read on the one thread that uses it.
    let validated = unsafe { pipeline.shaderValidation() } == MTLShaderValidation::Enabled;
    if !validated && pipeline.staticThreadgroupMemoryLength() > case.tg_memory {
        return Err(format!(
            "static threadgroup memory {}, the launch table's {}",
            pipeline.staticThreadgroupMemoryLength(),
            case.tg_memory
        ));
    }
    Ok((pipeline, validated))
}

/// What a completed command buffer logged. Shader validation reports an
/// invalid load or store there alone: the status stays Completed, with no
/// error.
fn logs(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Vec<String> {
    let container = cb.logs();
    let mut state = NSFastEnumerationState {
        state: 0,
        itemsPtr: std::ptr::null_mut(),
        mutationsPtr: std::ptr::null_mut(),
        extra: [0; 5],
    };
    let mut batch = [std::ptr::null_mut::<AnyObject>(); 16];
    let mut entries = Vec::new();
    loop {
        // SAFETY: the state and the batch of 16 are this function's own,
        // and the call points the state's items at at most that many logs.
        let n = unsafe {
            container.countByEnumeratingWithState_objects_count(
                NonNull::from(&mut state),
                NonNull::from(&mut batch).cast(),
                batch.len(),
            )
        };
        if n == 0 {
            return entries;
        }
        for i in 0..n {
            // SAFETY: the n items are logs the container holds, and the
            // command buffer has completed, so it doesn't change.
            let log: &AnyObject = unsafe { &**state.itemsPtr.add(i) };
            // SAFETY: description returns an NSString.
            let text: Retained<NSString> = unsafe { msg_send![log, description] };
            entries.push(text.to_string());
        }
    }
}

/// Fills the case's buffers and tables, launches it and waits for it.
fn launch(
    gpu: &Gpu,
    case: &Case,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<Memory, String> {
    let params = case.params.as_ref().map(|(i, file)| (*i, read(file)));
    if params.as_ref().is_some_and(|(_, p)| p.len() > 4096) {
        return Err("a parameter struct over setBytes' 4 KB".to_owned());
    }
    let input = |a: &Array| {
        let contents = a.input.as_deref().map(|p| read_exactly(p, a.bytes));
        buffer(gpu, a.bytes, contents.as_deref())
    };
    let memory = Memory {
        buffers: case.buffers.iter().map(|(_, _, a)| input(a)).collect(),
        tables: case
            .tables
            .iter()
            .map(|(_, t)| {
                let arrays: BTreeMap<usize, _> =
                    t.arrays.iter().map(|(e, a)| (*e, input(a))).collect();
                // Unbound entries point nowhere the kernel may go: 0.
                let mut addresses = vec![0u8; t.entries * 8];
                for (e, b) in &arrays {
                    addresses[e * 8..e * 8 + 8].copy_from_slice(&b.gpuAddress().to_le_bytes());
                }
                (buffer(gpu, addresses.len(), Some(&addresses)), arrays)
            })
            .collect(),
    };
    let cb = gpu.queue.commandBuffer().ok_or("no command buffer")?;
    let enc = cb.computeCommandEncoder().ok_or("no encoder")?;
    enc.setComputePipelineState(pipeline);
    for ((i, _, _), b) in case.buffers.iter().zip(&memory.buffers) {
        // SAFETY: the index is the lowering's binding of this buffer, which
        // holds the case's elements of its type from offset 0. The CPU
        // wrote it before the commit and reads it only after
        // waitUntilCompleted, and `memory`, which owns it, outlives that.
        unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, *i) };
    }
    for ((_, t), (addresses, arrays)) in case.tables.iter().zip(&memory.tables) {
        // SAFETY: as above, the table's binding, its addresses those of
        // arrays `memory` owns too.
        unsafe { enc.setBuffer_offset_atIndex(Some(addresses), 0, t.index) };
        let usage = if t.write {
            MTLResourceUsage::Read | MTLResourceUsage::Write
        } else {
            MTLResourceUsage::Read
        };
        for b in arrays.values() {
            // The arrays a table points into, resident (FuryKernel.Metal).
            enc.useResource_usage(ProtocolObject::from_ref(&**b), usage);
        }
    }
    if let Some((i, p)) = &params {
        // SAFETY: setBytes copies the struct's bytes before it returns.
        unsafe { enc.setBytes_length_atIndex(NonNull::from(p.as_slice()).cast(), p.len(), *i) };
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: case.groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: case.threads,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    if cb.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "the command buffer: {:?}",
            cb.error().map(|e| e.localizedDescription().to_string())
        ));
    }
    let logged = logs(&cb);
    if !logged.is_empty() {
        return Err(format!("the command buffer logged: {}", logged.join("; ")));
    }
    Ok(memory)
}

/// An output against the interpreter's, under its rule: the elements
/// compared, or how many differ and the first of them.
fn compare(
    what: &str,
    array: &Array,
    buf: &ProtocolObject<dyn MTLBuffer>,
) -> Result<usize, String> {
    let Some((want_file, alt_file, rule)) = &array.output else {
        return Ok(0);
    };
    let ty = array.ty;
    let count = array.bytes / ty.size();
    let got = contents(buf, array.bytes);
    let want = read_exactly(want_file, array.bytes);
    let alt = alt_file.as_deref().map(|p| read_exactly(p, array.bytes));
    let bounds = rule.bound.as_deref().map(|p| read_exactly(p, count * 8));
    let bad: Vec<usize> = (0..count)
        .filter(|&i| {
            let bound = bounds.as_deref().map(|bytes| {
                f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]))
            });
            let alt = alt.as_deref().map(|alt| bits(alt, ty, i));
            !agrees(ty, rule, bits(&got, ty, i), bits(&want, ty, i), alt, bound)
        })
        .collect();
    match bad.first() {
        None => Ok(count),
        Some(&i) => Err(format!(
            "{what}: {} of {count} differ, the first [{i}] {:#x}, the interpreter's {:#x}",
            bad.len(),
            bits(&got, ty, i),
            bits(&want, ty, i)
        )),
    }
}

/// A case's elements compared, and whether Metal validated its shaders.
fn run_case(gpu: &Gpu, case: &Case) -> Result<(usize, bool), String> {
    let (pipeline, validated) = prepare(gpu, case)?;
    // A grid of no threadgroups does nothing: a runtime skips it.
    if case.groups == 0 {
        return Ok((0, validated));
    }
    let memory = launch(gpu, case, &pipeline)?;
    let mut compared = 0;
    let mut errors = Vec::new();
    let mut tally = |result| match result {
        Ok(count) => compared += count,
        Err(e) => errors.push(e),
    };
    for ((_, name, array), buf) in case.buffers.iter().zip(&memory.buffers) {
        tally(compare(name, array, buf));
    }
    for ((name, t), (_, arrays)) in case.tables.iter().zip(&memory.tables) {
        for (e, array) in &t.arrays {
            tally(compare(&format!("{name}[{e}]"), array, &arrays[e]));
        }
    }
    if errors.is_empty() {
        Ok((compared, validated))
    } else {
        Err(errors.join("; "))
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let (default_options, root) = match args.as_slice() {
        [flag, root] if flag == "--default-options" => (true, PathBuf::from(root)),
        [root] if !root.starts_with("--") => (false, PathBuf::from(root)),
        _ => fail("usage: gpu-conformance [--default-options] CORPUS"),
    };
    let device = MTLCreateSystemDefaultDevice().unwrap_or_else(|| fail("no Metal device"));
    // The corpus is lowered for AppleGPU9: its limits, SIMD width and
    // fragments are GPU family 9's.
    if !device.supportsFamily(MTLGPUFamily::Apple9) {
        fail(&format!("{} is not of GPU family Apple9", device.name()));
    }
    let queue = device
        .newCommandQueue()
        .unwrap_or_else(|| fail("no command queue"));
    let gpu = Gpu {
        device,
        queue,
        default_options,
    };
    println!("GPU: {}", gpu.device.name());
    println!(
        "macOS: {}",
        NSProcessInfo::processInfo().operatingSystemVersionString()
    );
    println!(
        "compile options: {}",
        if default_options {
            "Metal's defaults (MSL 4.1, fast math)"
        } else {
            "the lowered ones (MSL 4.1, safe math)"
        }
    );
    println!(
        "MTL_SHADER_VALIDATION: {}",
        env::var("MTL_SHADER_VALIDATION").unwrap_or_else(|_| "unset".to_owned())
    );
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("cases"))
        .unwrap_or_else(|e| fail(&format!("{}: {e}", root.display())))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    dirs.sort();
    if dirs.is_empty() {
        fail(&format!("{}: no cases", root.join("cases").display()));
    }
    let (mut failed, mut validated) = (0, 0);
    for d in &dirs {
        let c = parse_case(d);
        // An autorelease pool per case: the command buffers and compiler
        // objects a case autoreleases are freed when it ends.
        let result = objc2::rc::autoreleasepool(|_| run_case(&gpu, &c));
        match result {
            Ok((n, v)) => {
                println!("ok   {} ({n} elements)", c.name);
                validated += usize::from(v);
            }
            Err(e) => {
                println!("FAIL {}: {e}", c.name);
                failed += 1;
            }
        }
    }
    println!(
        "Metal validated the shaders of {validated} of the {} cases that passed",
        dirs.len() - failed
    );
    println!("{} cases, {failed} failed", dirs.len());
    process::exit(i32::from(failed > 0));
}

// The comparison, which no GPU run can check: every rule against values
// that must agree and values that mustn't. Comparisons of floats are of
// powers of two, exact.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn rule(s: &str) -> Rule {
        Rule::parse(s, Path::new("."))
    }

    #[test]
    fn ulps_in_the_binade_below_a_power_of_two() {
        // 2.0: the unit of [1, 2), on both sides.
        assert_eq!(ulp_distance(Ty::F32, 0x3fff_fffc, 0x4000_0000), 4.0);
        assert_eq!(ulp_distance(Ty::F32, 0x4000_0002, 0x4000_0000), 4.0);
        // 1.5: its own.
        assert_eq!(ulp_distance(Ty::F32, 0x3fc0_0004, 0x3fc0_0000), 4.0);
        // The least normal and below it: the subnormals' unit.
        assert_eq!(ulp_distance(Ty::F32, 0x0080_0001, 0x0080_0000), 1.0);
        assert_eq!(ulp_distance(Ty::F32, 0x0000_0003, 0x0000_0001), 2.0);
        assert_eq!(ulp_distance(Ty::F16, 0x3bff, 0x3c00), 1.0);
        assert_eq!(ulp_distance(Ty::Bf16, 0x3f7f, 0x3f80), 1.0);
    }

    #[test]
    fn nans_by_class_or_whole() {
        let (bits, class, any) = (rule("bits"), rule("bits+class"), rule("bits+nan"));
        assert!(agrees(
            Ty::F32,
            &class,
            0xffc0_0001,
            0x7fc0_0000,
            None,
            None
        ));
        assert!(!agrees(
            Ty::F32,
            &class,
            0x7f80_0001,
            0x7fc0_0000,
            None,
            None
        ));
        assert!(agrees(Ty::F32, &any, 0x7f80_0001, 0x7fc0_0000, None, None));
        assert!(!agrees(
            Ty::F32,
            &bits,
            0xffc0_0000,
            0x7fc0_0000,
            None,
            None
        ));
        assert!(agrees(Ty::F16, &class, 0x7e01, 0xfe00, None, None));
        assert!(!agrees(Ty::Bf16, &class, 0x7f81, 0x7fc0, None, None));
    }

    #[test]
    fn subnormals_flushed_to_their_signed_zeros_only() {
        let ftz = rule("bits+ftz");
        assert!(agrees(Ty::F32, &ftz, 0x8000_0000, 0x8000_0001, None, None));
        assert!(!agrees(Ty::F32, &ftz, 0x0000_0000, 0x8000_0001, None, None));
        assert!(!agrees(Ty::F32, &ftz, 0x807f_ffff, 0x8000_0001, None, None));
        // Tininess before rounding: ±2^-126 as the zero of its sign.
        assert!(agrees(Ty::F32, &ftz, 0x0000_0000, 0x0080_0000, None, None));
        assert!(agrees(Ty::F32, &ftz, 0x8000_0000, 0x8080_0000, None, None));
        assert!(!agrees(Ty::F32, &ftz, 0x8000_0000, 0x0080_0000, None, None));
        assert!(!agrees(Ty::F32, &ftz, 0x0000_0000, 0x0080_0001, None, None));
        assert!(agrees(Ty::Bf16, &ftz, 0x8000, 0x8080, None, None));
        assert!(agrees(Ty::Bf16, &ftz, 0x8000, 0x8012, None, None));
        assert!(!agrees(Ty::Bf16, &ftz, 0x8013, 0x8012, None, None));
        // The flushed run, under ftz alone.
        assert!(agrees(
            Ty::F32,
            &ftz,
            0x3f80_0000,
            0x3f80_0001,
            Some(0x3f80_0000),
            None
        ));
        assert!(!agrees(
            Ty::F32,
            &rule("bits"),
            0x3f80_0000,
            0x3f80_0001,
            Some(0x3f80_0000),
            None
        ));
    }

    #[test]
    fn operands_returned_unflushed() {
        let (op, op_zeros) = (rule("bits+ftz+operand"), rule("bits+nan+ftz+operand+zeros"));
        assert!(agrees(
            Ty::F32,
            &op,
            0x0000_0005,
            0x3f80_0000,
            Some(0x0000_0000),
            None
        ));
        assert!(!agrees(
            Ty::F32,
            &op,
            0x8000_0005,
            0x3f80_0000,
            Some(0x0000_0000),
            None
        ));
        assert!(agrees(
            Ty::F32,
            &op_zeros,
            0x8000_0005,
            0x3f80_0000,
            Some(0x0000_0000),
            None
        ));
        // Against the flushed run only, and only under ftz.
        assert!(!agrees(Ty::F32, &op, 0x0000_0005, 0x0000_0000, None, None));
        let no_ftz = rule("bits+operand");
        assert!(!agrees(
            Ty::F32,
            &no_ftz,
            0x0000_0005,
            0x3f80_0000,
            Some(0x0000_0000),
            None
        ));
    }

    #[test]
    fn either_zero() {
        let zeros = rule("bits+zeros");
        assert!(agrees(
            Ty::F32,
            &zeros,
            0x8000_0000,
            0x0000_0000,
            None,
            None
        ));
        assert!(agrees(Ty::F16, &zeros, 0x0000, 0x8000, None, None));
        assert!(!agrees(
            Ty::F32,
            &zeros,
            0x0000_0001,
            0x0000_0000,
            None,
            None
        ));
    }

    #[test]
    fn ulps_and_bounds_but_infinities_only_as_themselves() {
        let (bits, ulp4) = (rule("bits"), rule("bits+ulp4"));
        assert!(agrees(Ty::F32, &ulp4, 0x3fff_fffc, 0x4000_0000, None, None));
        assert!(!agrees(
            Ty::F32,
            &ulp4,
            0x3fff_fffb,
            0x4000_0000,
            None,
            None
        ));
        assert!(agrees(
            Ty::F32,
            &bits,
            0x3f80_0000,
            0x3f80_0010,
            None,
            Some(1e-5)
        ));
        assert!(!agrees(
            Ty::F32,
            &bits,
            0x3f80_0000,
            0x3f80_0100,
            None,
            Some(1e-5)
        ));
        assert!(!agrees(
            Ty::F32,
            &bits,
            0x3f80_0000,
            0x7f80_0000,
            None,
            Some(f64::INFINITY)
        ));
        assert!(!agrees(
            Ty::F32,
            &bits,
            0x7f80_0000,
            0x3f80_0000,
            None,
            Some(f64::INFINITY)
        ));
        assert!(agrees(
            Ty::F32,
            &bits,
            0xff80_0000,
            0xff80_0000,
            None,
            Some(f64::INFINITY)
        ));
    }
}
