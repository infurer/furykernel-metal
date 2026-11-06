-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | furykernel-metal: the Apple Metal target of FuryKernel.

'metal' lowers checked kernels for 'AppleGPU9' to Metal Shading Language
and their launch tables ("FuryKernel.Backend"). Runtimes compile the source
with Metal at run time, and do what it leaves to them:

* run on macOS 27 or later, and compile with the options lowered with it
  ('FuryKernel.Backend.loweredOptions'): MSL 4.1, the floor, and
  'FuryKernel.Backend.Strict' arithmetic, which is @mathMode = .safe@
  with @mathFloatingPointFunctions = .precise@, or
  @fastMathEnabled = false@ (deprecated since macOS 15), which sets both;
  @mathMode@ alone leaves the functions fast. The source sets its math
  mode itself too (@#pragma METAL fp math_mode(safe)@), but binary16's
  minima and maxima, which have no precise overloads, rest on the
  options as well;
* leave every other @MTLCompileOptions@ property at its default:
  @floatingPointConversionRoundingMode@ toward zero truncates float's
  narrowing to half, and no pragma undoes it; @preprocessorMacros@ can
  redefine names the source prints; @requiredThreadsPerThreadgroup@ is
  one size for every kernel of the library;
* check the device, @supportsFamily(.apple9)@, and each pipeline:
  @maxTotalThreadsPerThreadgroup@ at least the kernel's
  'FuryKernel.Backend.lkThreads', and @threadExecutionWidth@ 32, the SIMD
  width the kernels are built for;
* leave threadgroup memory to the source, which declares the arrays
  statically: the kernels have no threadgroup arguments, so there is no
  @threadgroupMemoryLength@ to set. 'FuryKernel.Backend.lkTgMemory', the
  arrays' layout's total, is for information, and the most Metal
  allocates: it drops the arrays a kernel doesn't use. Under shader
  validation a pipeline reports about twice its threadgroup memory, and
  with API validation on too, a kernel of over 16 KiB of it is refused at
  dispatch;
* bind the buffers and tables at their indices in
  'FuryKernel.Backend.lkBindings', each buffer at an offset that is a
  multiple of its alignment in the ABI ('FuryKernel.Abi.bindingAlign', in
  'FuryKernel.Backend.lkAbi') and a table's buffer 8-byte aligned, and the
  parameter struct at 'FuryKernel.Backend.lkParamsBinding', with
  @setBytes@ (up to 4 KB) or from a buffer at an offset that is a
  multiple of 4;
* point each table entry at an array aligned to its table's alignment
  ('FuryKernel.Abi.tableBindingAlign'), and make the arrays the entries
  point into resident: with @useResource@ (@.write@ for a table written
  through), or in a residency set. A pack is one load or store of up to
  16 bytes, and at an address that isn't a multiple of 4 it uses the
  address rounded down to a multiple of 4, with nothing reported, even
  under validation;
* order a dispatch that reads what an earlier one wrote: a serial compute
  encoder, the default, orders its dispatches, a concurrent one only with
  a memory barrier between them;
* dispatch 'FuryKernel.Backend.lkGroups' threadgroups of
  'FuryKernel.Backend.lkThreads' threads with @dispatchThreadgroups@, not
  @dispatchThreads@, whose grid needn't be a multiple of the threadgroup.
  Skip a launch whose grid comes to 0 threadgroups, which does nothing,
  and which Metal's API validation refuses. Refuse one of 2^32 threads or
  more, which Metal runs wrapped, without an error even under validation
  (2^22 threadgroups of 1024 threads run none of them); the core's gate
  also refuses a kernel whose grid comes to that for some legal value of
  its dimensions;
* repeat the GPU conformance run (@nix run .#gpu-conformance@, test/gpu)
  on each new GPU and each new macOS: the target relies on what MSL
  doesn't specify (@simd_sum@'s order, the fragments' lane layout), and
  works around Metal's miscompiles, as observed on one GPU and one macOS.

The list is Metal 3's: Metal 4's compute encoders order no dispatches
without barriers, and its argument tables have no @setBytes@, so a runtime
on them needs a list of its own. What the list says of Metal's behaviour
was observed on an M3 Max (AppleGPU9), macOS 27.2.
-}
module FuryKernel.Metal (
  AppleGPU9,
  metal,
  mslLanguage,
) where

import FuryKernel.Metal.Msl (metal, mslLanguage)
import FuryKernel.Metal.Target (AppleGPU9)
