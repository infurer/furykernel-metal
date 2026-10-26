-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | furykernel-metal: the Apple Metal target of FuryKernel.

The target descriptions, the lowering to Metal Shading Language and the
generated launchers are added module by module.
-}
module FuryKernel.Metal (
  AppleGPU9,
) where

import FuryKernel.Metal.Target (AppleGPU9)
