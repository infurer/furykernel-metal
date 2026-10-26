-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The Apple GPU families this library targets.

'AppleGPU9' is Apple's GPU family 9, of the A17 Pro, A18, M3 and M4, which
the first kernels are written and tuned for. What this library records as
observed on the GPU, it observed on one M3 Max, under macOS 27.2. Its
limits:

* SIMD groups of 32 lanes;
* up to 1024 threads per threadgroup, a multiple of 32;
* 32 KiB of threadgroup memory per threadgroup;
* 31 buffer bindings, the parameter struct included;
* lane loads and stores of up to 16 bytes.

Fragments (8×8 SIMD-group matrices), relaxed atomics only and no forward
progress between threadgroups come with the stages that need them; M5's
@AppleGPU10@, with its tensor ops, comes later.
-}
module FuryKernel.Metal.Target (
  AppleGPU9,
) where

import FuryKernel.Target (Target (..))

-- | Apple's GPU family 9: the A17 Pro, A18, M3 and M4.
data AppleGPU9

instance Target AppleGPU9 where
  type TargetName AppleGPU9 = "AppleGPU9"
  type SimdWidth AppleGPU9 = 32
  type MaxThreads AppleGPU9 = 1024
  type TgMemBytes AppleGPU9 = 32768
  type MaxBuffers AppleGPU9 = 31
  type LaneLoadMax AppleGPU9 = 16
