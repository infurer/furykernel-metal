-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | AppleGPU9's limits, as the rules read them.
module Test.FuryKernel.Metal.TargetTest (tests) where

import FuryKernel.Constraint (checkTgTotal, checkThreads, withThreads)
import FuryKernel.Finding (Finding (..), Rule (..))
import FuryKernel.Metal (AppleGPU9)
import FuryKernel.Target (TargetCaps (..), targetCaps)
import FuryKernel.Type (ScalarType (..))
import GHC.TypeNats (natVal)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Target"
    [ testCase "the limits" $
        targetCaps @AppleGPU9
          @?= TargetCaps
            { capName = "AppleGPU9"
            , capSimdWidth = 32
            , capMaxThreads = 1024
            , capTgMemBytes = 32768
            , capMaxBuffers = 31
            , capLaneLoadMax = 16
            , capFragments = [TBF16, TF16, TF32]
            }
    , testCase "the rules, in AppleGPU9's words" $ do
        withThreads @AppleGPU9 1024 natVal @?= Right 1024
        checkThreads (targetCaps @AppleGPU9) 1056
          @?= [Finding Threads "Threads per threadgroup on AppleGPU9: 1056; the limit is 1024."]
        checkTgTotal (targetCaps @AppleGPU9) [("partial", 4096), ("tile", 32768)]
          @?= [ Finding
                  ThreadgroupMemory
                  "Threadgroup memory on AppleGPU9, each array taking its size rounded up to its alignment: partial 4096 B + tile 32768 B = 36864 B; the limit is 32768 B."
              ]
    ]
