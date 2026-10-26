-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

module Main (main) where

import Test.FuryKernel.Metal.MslTest qualified as MslTest
import Test.FuryKernel.Metal.TargetTest qualified as TargetTest
import Test.Tasty (defaultMain, testGroup)

main :: IO ()
main =
  defaultMain $
    testGroup
      "furykernel-metal"
      [ TargetTest.tests
      , MslTest.tests
      ]
