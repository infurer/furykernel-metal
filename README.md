<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# furykernel-metal

furykernel-metal is FuryKernel's Apple Metal target: a Haskell library that
lowers the core's checked kernels to Metal Shading Language. The core,
furykernel, is at <https://github.com/infurer/furykernel>. It holds:

- the target descriptions (`FuryKernel.Metal.Target`): `AppleGPU9`, Apple's
  GPU family 9 of the A17 Pro, A18, M3 and M4, with the limits the core's
  gate holds kernels to; M1's `AppleGPU7` and M5's `AppleGPU10` come
  later;
- the backend, `metal` (`FuryKernel.Metal.Msl`): a set of checked kernels
  as one MSL 4.1 translation unit (macOS 27, the floor), the options a
  runtime compiles it with, and each kernel's launch table: its entry
  point, bindings, geometry and ABI.

It emits source only. Runtimes compile it with Metal at run time, on
macOS 27 or later, so the library builds and is tested on any system, and
golden tests pin its output. What a runtime does with the source and the
launch tables is in `FuryKernel.Metal`'s documentation.

## Status

Version 0.2, in development, with the core: the lowering covers what the
first kernels use, packs, tables and SIMD-group matrices included. The
lowering of FuryKernel programs into launch schedules comes later. What
the documentation records of Metal's behaviour was observed on one M3 Max,
under macOS 27.2.

## Building and testing

```sh
nix build                  # the library, its test suite run in the check phase
nix develop                # GHC, cabal and the tools
cabal build                # inside the dev shell
nix flake check            # the build, its tests and every other check
nix run .#gpu-conformance  # on an Apple9 Mac: the GPU conformance run
```

The GPU conformance run needs a Mac whose GPU is of family Apple9, on
macOS 27 or later, so it runs outside CI: before a merge request, and
again on each new GPU or macOS. It lowers a corpus of kernels
(`test/gpu`), runs it on the GPU three times, as a runtime would, under
shader validation and with Metal's default options, and holds every
output to the core's interpreter by what the IR leaves a target.

## License

BSD 3-Clause, in `LICENSES/BSD-3-Clause.txt`.
