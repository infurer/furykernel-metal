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
  statically: 'lkTgMemory', what they take, is for information, and a
  runtime that sets @threadgroupMemoryLength@ too allocates it twice;
* bind the buffers and tables at their indices in 'lkBindings', and the
  parameter struct at 'lkParamsBinding', with @setBytes@ (up to 4 KB) or
  from a buffer at an offset that is a multiple of 4;
* bind a table's buffer 8-byte aligned, and make the arrays its entries
  point into resident (@useResource@);
* dispatch 'lkGroups' threadgroups of 'lkThreads' threads with
  @dispatchThreadgroups@, not @dispatchThreads@, whose grid needn't be a
  multiple of the threadgroup, and skip a launch whose grid comes to 0
  threadgroups, which does nothing and which Metal refuses.
-}
module FuryKernel.Metal (
  AppleGPU9,
  metal,
  mslLanguage,
) where

import FuryKernel.Metal.Msl (metal, mslLanguage)
import FuryKernel.Metal.Target (AppleGPU9)
