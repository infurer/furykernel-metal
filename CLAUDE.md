<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

furykernel-metal is FuryKernel's Apple Metal target: a library that
implements the core's target interface for Apple GPUs. It holds the target
descriptions (`AppleGPU9`, Apple's GPU family 9 of the A17 Pro, A18, M3
and M4, first; `AppleGPU7` for M1 and `AppleGPU10` for M5 later) and the
lowering of checked kernels to Metal Shading Language, with the launch
tables runtimes read. Runtimes write their launchers themselves, from
those tables and the core's `FuryKernel.Emit.Rust`. The GPU lowering of
FuryKernel programs into launch schedules is planned. It depends only on
the core library, furykernel.

The library never runs anything on a GPU itself: it emits MSL source,
which the consuming runtime compiles with Metal at run time, on macOS 27
or later. So it builds and is tested on any system, and golden tests pin
its output; the GPU conformance run (`test/gpu/`) runs that output on an
Apple9 Mac's GPU. What the repository records of Metal's behaviour was
observed on one M3 Max, under macOS 27.2.

Like the core, it knows no models: model kernels live with the runtime that
runs them.

## Commands

```sh
nix develop                # GHC, cabal, ghcid, hlint, cabal2nix, reuse
cabal build                # inside the dev shell
nix fmt
nix flake check --all-systems --no-update-lock-file
nix run .#gpu-conformance  # on an Apple9 Mac: the corpus on its GPU
```

After changing `furykernel-metal.cabal`, regenerate `furykernel-metal.nix`:

```sh
cabal2nix . | sed 's|^  src = .*;$|  src = ./.;|' > furykernel-metal.nix
```

`nix flake check` evaluates every system but builds only the current
system's checks. The per-commit gate builds them for all systems; on a Mac
the Linux ones go to its remote builder. Flakes only see files git tracks,
so `git add` new files first:

```sh
nix fmt && git diff --exit-code &&
  nix flake check --all-systems --no-update-lock-file &&
  (
    for s in x86_64-linux aarch64-linux aarch64-darwin; do
      nix build --no-link --impure --expr \
        "builtins.attrValues (builtins.getFlake \"git+file://$PWD\").checks.$s" || exit 1
    done
  )
```

The GPU conformance run needs an Apple9 GPU, so it is no check and runs
outside CI: before a merge request, on an Apple9 Mac, with every case
passing, and again on each new GPU or macOS, as `FuryKernel.Metal` asks
of runtimes.

## Architecture

- `flake.nix`: flake-parts on nixos-26.05; systems x86_64-linux,
  aarch64-linux and aarch64-darwin. The `furykernel` input is the core,
  whose package `nix/package.nix` builds this one against. `flake.lock`
  pins it as a `github:infurer/furykernel` revision on its master, as
  `nix run .#check-pins` checks.
- `furykernel-metal.cabal`, `src/`: the library, one package.
  `furykernel-metal.nix` is cabal2nix's output, committed so that
  evaluation never builds anything first.
  - `FuryKernel.Metal.Target`: the GPU families, `AppleGPU9` (family 9) so
    far, as `Target` instances; its fragments are bfloat, half and float.
  - `FuryKernel.Metal.Msl`: `metal`, the backend: checked kernels of
    distinct names as one MSL 4.1 translation unit (two of one name are
    refused, as the CPU backend refuses them), which a runtime compiles as
    its options say (`loweredOptions`: MSL 4.1, macOS 27's, the floor;
    `Strict` arithmetic), and their launch tables. The source's prelude
    refuses MSL before 4.1 (`#if __METAL_VERSION__ < 410`, an `#error`),
    sets its own math mode (`#pragma METAL fp math_mode(safe)`) and, after
    it, as safe math mode sets the contract on,
    `#pragma METAL fp contract(off)`: no multiplication and addition
    contract within a statement, and a multiply-add is fused only where a
    kernel writes `fmaE`, as `metal::fma`. Each parameter struct's size
    and its fields' offsets are asserted against the ABI's
    (`static_assert`). The IR leaves binary32 subnormals to the target, and
    MSL 4.1's section 8.1 lets them be flushed to zero; on the M3 Max,
    float arithmetic flushes subnormal operands and results to zeros of
    their signs (fma's results tiny before rounding too), comparisons see
    them as zeros, and so do fmin, fmax and select's condition, which
    return operands unflushed, simd_max over subnormal lanes gives lane
    0's, loads, stores, bfloat's widening and tanh keep the bits, and half
    keeps its subnormals, as the IR requires, though MSL lets them be
    flushed too: the target meets the IR there by observation alone. A
    converted NaN is the target's too: on the M3 Max, bfloat's widening
    keeps a signalling NaN signalling, half's widening and float's
    narrowing to half give the canonical NaN, and no NaN becomes an
    infinity (the narrowing to bfloat, by bits, quiets a NaN, keeping its
    sign and the top of its payload).
    Library calls are fully qualified and generated names start with
    `fk_`; floats print exactly; transcendental functions are
    `metal::precise::`, so results don't depend on the compile options'
    math mode, within the bounds of MSL 4.1's Table 8.1 (exp and log 4
    ulp, tanh 5, sqrt, rsqrt and fma correctly rounded); binary16 math and
    division go through binary32, rounding once more. Where macOS 27.2's
    compiler miscompiled the printing's MSL, the printing works around it,
    and `test/gpu/` holds each one's repro: a bf16 value a select, a var or
    a literal gives is held as its bits (a `ushort`), and every narrowing
    to bf16 goes through `fk_bf16`, by bits; u32 division by a divisor
    that isn't a literal goes through `fk_udiv` and `fk_urem`; binary16
    division doubles its operands first; float's sqrt and rsqrt, and
    float's narrowing to half, take their operands through `fk_keep`. A
    half select goes on its bits, which Metal's default options would make
    an fmin. A pack is one `ushort`, `ushort2` (two 16-bit
    elements), `uint`, `uint2` or `uint4` load or store through the
    buffer's pointer, taken apart and put together with `as_type`; a pack
    of four-byte elements stored to a threadgroup array is stored as their
    own vector (`float2`, `float4`, `int4` and the like; narrower ones as
    words). Where the kernel writes the memory (a writable buffer, a
    threadgroup array, a table written through), those words are named
    through typedefs that may alias every type (`typedef uint4 __attribute__((may_alias)) fk_uint4;` and the like), at every
    language version: as observed on macOS 27.2's compiler at MSL 4.1,
    type-based alias analysis otherwise takes a `uint4` access and a
    `float` one, say, to be of different objects, so that a pack reloaded
    after an element's store, or an element after a pack's, reads the old
    value (at 3.1 it takes vectors, though not scalars, to alias every
    type). A pack of memory the kernel only reads keeps the plain type:
    bindings never overlap, so no store can touch it.
    A SIMD-group barrier is `metal::simdgroup_barrier` over threadgroup
    memory, which only the golden tests pin: on the M3 Max, whose SIMD
    groups seem to run in lockstep, every case passes without it.
    A fragment is widened a lane's elements at a time
    (`thread_elements()`), and its rows scaled the same way, each lane's
    row found from the 8×8 layout observed on an M3 Max (AppleGPU9),
    which MSL doesn't specify (a lane holds two elements of row
    `(lane / 4 & 4) + (lane / 2 % 4)`, and a bfloat or half fragment's
    element k where a float fragment's is). A table
    is a `device const ulong*` of GPU addresses, each cast to a pointer
    where it is read or stored through, fragments loaded and stored
    through it too; the pointer is `device const` where the kernel only
    reads through the table. A fragment is a `metal::simdgroup_matrix` of 8×8,
    made zero, loaded and stored from a pointer to its first element with
    the stride as elements per row, and multiplied with
    `simdgroup_multiply_accumulate`; the GPU adds a product's terms in an
    order of its own, so such kernels match the interpreter to f32
    accumulation's bound, not bit for bit. Buffers bind from 0 in
    declaration order, tables after them, the parameter struct last.
    Every threadgroup array is declared `alignas(16)`, so Metal starts each
    at a multiple of 16 bytes; it allocates at most each one's size rounded
    up to 16, summed, the launch table's `lkTgMemory`: as observed on the
    M3 Max, that where the kernel uses every array, less where it doesn't,
    Metal dropping the arrays a kernel doesn't use and shrinking some indexed
    by constants. i32 addition, subtraction, multiplication, negation and
    magnitude go through uint, so that they wrap as the IR defines them,
    the magnitude in `fk_abs_i32`, which takes its operand once. half's
    negation and magnitude go through its bits: Metal's give a NaN the
    canonical one (observed on an M3 Max, AppleGPU9). float's magnitude
    goes through its bits too: macOS 27.2's compiler drops `metal::abs`
    where it takes the operand for non-negative (`abs(rsqrt(-0))` came out
    `-inf`).
    Float minima and maxima, and the SIMD-group reductions over them, are
    Metal's own (`metal::precise::fmin` and `metal::precise::fmax` in f32,
    whatever the compile options; `metal::fmin` and `metal::fmax` in f16,
    which has no precise overloads, held by the source's math mode and the
    lowered options; `metal::simd_min`, `metal::simd_max`): on the M3 Max
    fmin and fmax ignore a NaN operand, as the IR's minimumNumber does.
    Which of operands that compare equal (-0 and +0) or of two NaNs comes
    out is the target's choice, as the IR leaves it: MSL's section 6.6
    gives the first of equal operands, a call alone in a kernel gave the
    second on the M3 Max, and the compiler took `fmin(a, b)` and
    `fmin(b, a)` in one kernel for one call. Of
    the reductions, only simd_max over subnormal lanes is recorded (lane
    0's); what they choose among equal lanes, -0 and +0, or NaNs isn't.
    SIMD-group sums are `metal::simd_sum`, which adds pairwise, neighbours
    first, as the IR orders them (bit for bit on the M3 Max, in f32 away
    from binary32 subnormals, which it flushes, and in f16). It stays
    native: an explicit `simd_shuffle_xor` butterfly gives the same bits
    but was slower (by 0.47 % in a model's decoding at batch 1), so the
    order rests on the observation, which the corpus guards (a reversed or
    sequential order fails `flow`, `simd_sums` and `sum_rows`). And
    and Or evaluate both operands (`bool(int(a) & int(b))`); an integer
    narrows clamped with `metal::clamp` or `metal::min`, or wrapped
    through `uint` and `as_type`.
- `test/`: the test suite, tasty: the limits of the targets, and golden MSL
  of an RMSNorm kernel's translation unit and of every construct the
  printing fixes (packs of every container, through the typedefs that may
  alias, to and from threadgroup memory; tables read and written through;
  fragments; SIMD-group barriers; the arithmetic, conversions, literals
  and comparisons; signatures and kernel order), the launch tables of the
  RMSNorm kernel and of a table's bindings, threadgroup memory as they
  report it, and the refusal of two kernels of one name.
- `test/gpu/`: the GPU conformance run (`apps.gpu-conformance`).
  `Cases.hs` writes its corpus: the core's checks.cpu kernels, the core's
  test kernels and this library's own constructs (every pack container
  against element accesses and threadgroup memory, SIMD-group barriers,
  the target's limits, every binary16 and bfloat16), each lowered by
  `metal`, with the interpreter's inputs and outputs, each run twice with
  its write-only arrays poisoned (0xff and 0x5a bytes), and a rule for each
  output, the IR's leave there: arithmetic's NaNs by class (quiet or
  signalling), conversions', minima's and maxima's whole; binary32
  subnormals as signed zeros where the IR lets a target flush them (an
  interpreter run on flushed inputs, parameters and literals), results
  tiny before rounding that round to the least normal too, an operand
  returned unflushed; either zero from minima and maxima; MSL's ulp bounds
  for exp, log and tanh; products of fragments within (k + 2) · 2^-24 of
  the sum of their products' magnitudes and the least normal. The runner,
  Rust on objc2-metal, refuses a GPU outside family Apple9, compiles each
  case with the lowered options, checks its pipeline (threads, SIMD width,
  threadgroup memory at most the launch table's where Metal doesn't
  validate it), binds, dispatches and compares, failing a case on anything
  the command buffer logged; the comparison is unit-tested
  (`checks.gpu-runner-tests`). It
  runs three times: as a runtime runs it, under shader validation (whose
  invalid loads and stores only the logs report), and with Metal's
  default options (`--default-options`, fast math), so that the source's
  pragmas and precise functions must hold; and it prints the GPU, macOS
  and the compiler.
- `nix/`: `package.nix` (the library, built with GHC 9.12 under
  `-Werror`), `checks.nix` (the build with its tests, hlint,
  cabal2nix-in-sync and `reuse lint`), `devshell.nix` (`checks.devshell`
  builds its shell), `pins.nix` (`nix run .#check-pins`), `rust.nix`
  (crane and the Rust toolchain for the runner, and rustfmt), `gpu.nix`
  (the corpus, `checks.gpu-corpus` on every system; the runner, the app,
  `checks.gpu-runner-clippy` and `checks.gpu-runner-tests` on
  aarch64-darwin) and `treefmt.nix` (`nix fmt` and `checks.treefmt`).
- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `checks` runs
  `nix flake check --all-systems`, which builds x86_64-linux's checks, and
  builds the packages of x86_64-linux and aarch64-linux, whose tests run in
  their checkPhase; `pins` runs `nix run .#check-pins` (`nix/pins.nix`):
  every input of ours locked at a commit on GitHub master, so that it fails
  until a branch's dependencies are merged, published and locked there;
  `publish-github`, on master once those passed, pushes master, and the
  tags it holds, to the public GitHub repository.

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `BSD-3-Clause`). Files that can't hold comments are
  annotated in `REUSE.toml`.
- Every file type has a formatter in `nix/treefmt.nix`, enabled in the
  commit that adds the first file of that type. Haskell is formatted by
  fourmolu (`fourmolu.yaml`), cabal files by cabal-gild.
- GHC 9.12 (`haskell.packages.ghc912`, set once in `nix/package.nix`),
  GHC2024 and the warnings in `furykernel-metal.cabal`'s `warnings` stanza;
  `-Werror` only in the Nix build, so work in progress still loads in ghci.
  Modules have export lists. `-Wmissing-home-modules` makes a module the
  .cabal file doesn't list, which Hackage's source tarball would lack, an
  error.
- Every dependency has a lower and an upper bound by the PVP
  (https://pvp.haskell.org/), below its next major version, stated once:
  the test suite shares the library's bounds and states only its own
  dependencies'. base stays below the next GHC's until that GHC is tested.
- `CHANGELOG.md` is the change log Hackage shows on the package's page, a
  section per version; 0.2.0.0 is the initial release.
- Nothing host-specific in the repository: no hostnames, absolute paths of
  anyone's machine, or names of private repositories and services. master
  is mirrored to a public GitHub repository in full.

## Git workflow

- One commit per reasonably sized feature. Docs (README, this file) change
  in the same commit as the feature they describe.
- Subjects in nixpkgs style, `<attr or area>: <summary>`. Short bodies: why,
  and facts that aren't obvious from the diff.
- Commit with `git commit -s`, and with an `Assisted-by:` trailer naming
  the AI assistant that helped, such as `Assisted-by: Claude Opus 5.5`,
  directly above the Signed-off-by. No other trailers.
- `flake.lock` changes only together with the input change that needs them.
- Every commit passes the per-commit gate (see Commands).
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- `path:` flake inputs, or `flake.lock` updates riding along in unrelated
  commits.
- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
