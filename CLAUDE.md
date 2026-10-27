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

It never runs anything on a GPU: it emits MSL source, which the consuming
runtime compiles with Metal at run time. So it builds and is tested on any
system, and golden tests pin its output.

Like the core, it knows no models: model kernels live with the runtime that
runs them.

## Commands

```sh
nix develop                # GHC, cabal, ghcid, hlint, cabal2nix, reuse
cabal build                # inside the dev shell
nix fmt
nix flake check --all-systems --no-update-lock-file
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
    their signs, comparisons see them as zeros, and so do fmin, fmax and
    select's condition, which return operands unflushed, simd_max over
    subnormal lanes gives lane 0's, loads, stores, bfloat's widening and
    tanh keep the bits, and half keeps its subnormals. A converted NaN is
    the target's too: on the M3 Max, bfloat's widening keeps a signalling
    NaN signalling, half's widening and float's narrowing give the
    canonical NaN, and no NaN becomes an infinity. Library calls are
    fully qualified and generated names start with `fk_`; floats print
    exactly; transcendental functions are `metal::precise::`, so results
    don't depend on the compile options' math mode, within the bounds of
    MSL 4.1's Table 8.1 (exp and log 4 ulp, tanh 5, sqrt, rsqrt and fma
    correctly rounded); binary16 math and division go through binary32,
    rounding once more. A pack is one
    `ushort`, `ushort2` (two 16-bit elements, which a `uint` would alias),
    `uint`, `uint2` or `uint4` load or store through the
    buffer's pointer, taken apart and put together with `as_type`; a pack
    of four-byte elements stored to a threadgroup array is stored as their
    own vector (`float2`, `float4`, `int4` and the like; narrower ones as
    words). A table
    is a `device const ulong*` of GPU addresses, each cast to a pointer
    where it is read. A fragment is a `metal::simdgroup_matrix` of 8×8,
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
    from binary32 subnormals, which it flushes, and in f16). And
    and Or evaluate both operands (`bool(int(a) & int(b))`); an integer
    narrows clamped with `metal::clamp` or `metal::min`, or wrapped
    through `uint` and `as_type`.
    It refuses what it doesn't print yet, a `Target` finding for each:
    SIMD-group barriers, fragments widened into f32, fragments' rows scaled,
    tables written through and fragments loaded or stored through tables.
- `test/`: the test suite, tasty: the limits of the targets, and golden MSL,
  the launch tables of an RMSNorm kernel and of a table's bindings with it,
  packs through ushort, ushort2, uint2 and uint4 and to threadgroup arrays
  as float4, tables and fragments included.
- `nix/`: `package.nix` (the library, built with GHC 9.12 under
  `-Werror`), `checks.nix` (the build with its tests, hlint,
  cabal2nix-in-sync and `reuse lint`), `devshell.nix` (`checks.devshell`
  builds its shell), `pins.nix` (`nix run .#check-pins`) and `treefmt.nix`
  (`nix fmt` and `checks.treefmt`).
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
  Modules have export lists.
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
