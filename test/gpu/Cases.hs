-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

{- | apps.gpu-conformance's corpus: the core's checks.cpu kernels, the core's
test kernels and this library's own constructs, lowered by 'metal', each
with the interpreter's inputs and outputs. For each run, @cases/NAME@ holds
the translation unit (@source.metal@), the launch (@case.txt@: entry point,
threads, grid, threadgroup memory, bindings, tables, the parameter struct's
bytes) and the buffers' and tables' bits, which the runner in this
directory launches on the GPU and compares. Every run comes twice, its
write-only arrays poisoned with 0xff bytes (@NAME_I_ff@) and with 0x5a
bytes (@NAME_I_5a@), in the interpreter's run as on the GPU: an element
no store reaches keeps the poison, and one the GPU misses differs from the
interpreter's in one of the two.

The comparison is the IR's leave to a target, an output's rule:

* @bits@: bit for bit;
* @class@: a NaN out of arithmetic by its class, quiet or signalling, its
  sign and payload the target's, as the core's checks.cpu compares them;
* @nan@: any NaN for a NaN, where the IR leaves it whole to the target:
  conversions, minima and maxima;
* @ftz@: binary32 and bfloat16 subnormals as zeros of their signs: in
  operands, a second run of the interpreter on flushed inputs, parameters
  and literals, and in results, a subnormal's signed zero, or the least
  normal's, tininess being detected before rounding;
* @operand@: an operand returned (select, minima, maxima), under @ftz@
  unflushed where the flushed run gives the zero of its sign;
* @zeros@: of -0 and +0, either (minima and maxima);
* @ulpN@: within N units in the last place, MSL's bound for precise
  transcendental functions (Table 8.1), at a power of two the unit of the
  binade below it;
* @bound:FILE@: within each element's bound, for products of fragments,
  which the GPU sums in an order of its own: (k + 2) · 2^-24 times the sum
  of the products' magnitudes (a third run, on the inputs' magnitudes),
  and the least normal, 2^-126.
-}
module Main (main) where

import Control.Monad (forM, forM_)
import Data.Bits (shiftR, xor, (.&.), (.|.))
import Data.ByteString qualified as BS
import Data.Int (Int32, Int8)
import Data.List (nub, sort)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Proxy (Proxy (..))
import Data.Text (Text)
import Data.Text qualified as T
import Data.Text.IO qualified as T
import Data.Word (Word16, Word32, Word64, Word8)
import FuryKernel
import FuryKernel.Abi (Abi (..), Field (..), abiOf)
import FuryKernel.Backend (Backend (..), Lowered (..), LoweredKernel (..))
import FuryKernel.Check (checkedIR)
import FuryKernel.IR (Expr (..), Literal (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Interp (Value (..), launch, runIRWithTables, withSeed, withTables)
import FuryKernel.Metal (AppleGPU9, metal)
import FuryKernel.Numerics (narrowBF16, narrowF16, widenF16)
import FuryKernel.Target (targetCaps)
import FuryKernel.Type (ScalarType (..), scalarSize)
import GHC.Float (castDoubleToWord64, castFloatToWord32, castWord32ToFloat)
import GHC.TypeNats (KnownNat, natVal)
import Numeric (showHex)
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import Test.FuryKernel.Gemv (GemvShape (..), GemvVariant (..), gemv)
import Test.FuryKernel.Kernels (rmsnorm, scale, scalePacked, throughBlocks)

------------------------------------------------------------------------------
-- Cases

-- | A kernel, the launches to compare it on, and what the IR leaves a target in its outputs.
data Case = Case
  { caseId :: Text
  , caseKernel :: Checked AppleGPU9
  , caseRuns :: [Inputs]
  , caseLeaves :: [([Name], [Leave])]
  -- ^ Outputs and their leaves; an output none names is compared bit for bit.
  }

-- | What the IR leaves a target in an output, beyond its bits: its rule's parts.
data Leave
  = -- | A NaN out of arithmetic: its sign and payload, not its class, the target's.
    NanClass
  | -- | A NaN the IR leaves whole to the target: a conversion's, a minimum's or a maximum's.
    AnyNan
  | -- | binary32 and bfloat16 subnormals as zeros of their signs, in operands and in results.
    Ftz
  | -- | An operand returned (select, minima, maxima): under 'Ftz', as it was.
    Operand
  | -- | Of -0 and +0, either.
    Zeros
  | -- | Within this many units in the last place.
    Ulp Int
  deriving stock (Eq, Ord)

data Inputs = Inputs
  { inParams :: [(Name, Value)]
  , inBuffers :: [(Name, [Value])]
  , inTables :: [(Name, [[Value]])]
  }

checked :: Text -> Kernel AppleGPU9 -> Checked AppleGPU9
checked what k = either (\fs -> error (T.unpack what <> ": " <> show fs)) id (check k)

cases :: [Case]
cases =
  [ Case "sum_rows" (checked "sum_rows" sumRows) [Inputs [("n", u 3)] [("x", f32s 3 192), ("y", zeros 3)] [("rows", [f32s (10 + i) 64 | i <- [0 .. 2]])]] []
  , Case "rmsnorm" (checked "rmsnorm" (rmsnorm @AppleGPU9 1000)) [Inputs [("rows", u 3), ("eps", VF32 1e-6)] [("x", bf16s 1 3000), ("w", bf16s 2 1000), ("y", bf16Zeros 3000)] []] []
  , Case "scale" (checked "scale" (scale @AppleGPU9)) [Inputs [("n", u 300), ("s", VF32 0.7)] [("x", bf16s 4 300), ("y", bf16Zeros 300)] []] []
  , Case "scale_packed" (checked "scale_packed" (scalePacked @AppleGPU9)) [Inputs [("packs", u 37), ("s", VF32 (-1.3))] [("x", bf16s 5 296), ("y", bf16Zeros 296)] []] []
  , simdSumsCase
  , narrowingsCase
  , -- Four entries of a table of 4001, each 2^14 elements.
    Case
      "through_blocks"
      (checked "through_blocks" (throughBlocks @AppleGPU9 True))
      [ Inputs
          [("entries", u 4001)]
          [("ids", map VU32 [3, 7, 1000, 4000]), ("x", f32s 70 256), ("y", zeros 256)]
          [("blocks", [if e `elem` [3, 7, 1000, 4000] then f32s (71 + fromIntegral e) 16384 else [] | e <- [0 .. 4000 :: Word32]])]
      ]
      []
  , Case "gemm_via" (checked "gemm_via" gemmVia) [Inputs [("k", u 32)] [] [("ab", [[], bf16s 36 512, bf16s 37 512]), ("cs", [[], zeros 256])]] []
  ]
    <> zipWith gemvCase [0 :: Int ..] gemvs
    <> [ gemmCase "gemm_bf16" (checked "gemm_bf16" (gemm @BF16)) (bf16s 30) (bf16s 31)
       , gemmCase "gemm_f16" (checked "gemm_f16" (gemm @F16)) (f16s 30) (f16s 31)
       , gemmCase "gemm_f32" (checked "gemm_f32" (gemm @Float)) (f32s 30) (f32s 31)
       , gemmCase "staged" (checked "staged" staged) (bf16s 32) (bf16s 33)
       , gemmCase "scaled_rows" (checked "scaled_rows" scaledRows) (f32s 34) (bf16s 35)
       , tileCase "tiled" (checked "tiled" (tiled @1024 @1024 @4096 (GemmTile 64 64 16 4))) (70, 72, 48)
       , tileCase "tiled_narrow" (checked "tiled_narrow" (tiled @1024 @2048 @512 (GemmTile 32 64 32 1))) (70, 72, 64)
       , arithF32Case
       , arithF16Case
       , fmaF32Case
       , fmaF16Case
       , arithIntCase
       , convCase
       , Case "flow" (checked "flow" flow) [Inputs [("n", u 50)] [("x", f32s 60 50), ("cnt", [VU32 (fromIntegral (w `mod` 13)) | w <- randoms 61 288]), ("y", zeros 288), ("z", zeros 288), ("w", replicate 288 (VU32 0))] []] []
       , reductionsCase
       , paramsCase
       , rangedCase
       ]
 where
  gemvs =
    [ (GemvShape 64 256 1, GemvVariant 1 32 4 1)
    , (GemvShape 64 256 3, GemvVariant 2 64 8 2)
    , (GemvShape 128 512 2, GemvVariant 4 128 8 4)
    , (GemvShape 64 512 3, GemvVariant 1 256 8 8)
    , (GemvShape 32 1024 2, GemvVariant 8 128 8 4)
    ]
  gemvCase i (sh, v) =
    let name = "gemv_" <> T.pack (show i)
        k = either (error . show) id (gemv @AppleGPU9 name sh v)
        outs = fromIntegral sh.outputs
        ins = fromIntegral sh.inputs
        m = fromIntegral sh.batch
     in Case name (checked name k) [Inputs [] [("x", bf16s 20 (m * ins)), ("w", bf16s 21 (outs * ins)), ("y", bf16Zeros (m * outs))] []] []
  gemmCase name k a b = Case name k [Inputs [("k", u 64)] [("a", a (16 * 64)), ("b", b (16 * 64)), ("c", zeros 256)] []] []
  tileCase name k (m, n, kk) = Case name k [Inputs [("m", u m), ("n", u n), ("k", u kk)] [("a", bf16s 40 (m * kk)), ("b", bf16s 41 (n * kk)), ("c", zeros (m * n))] []] []

------------------------------------------------------------------------------
-- Kernels

-- | Each threadgroup's row of x against the array of the row's table entry, summed by the SIMD group.
sumRows :: Kernel AppleGPU9
sumRows = kernel "sum_rows" $ do
  n <- dim "n" (DimRange 1 16 1)
  x <- buffer @ReadOnly @16 @Float "x" (n * 64)
  y <- buffer @WriteOnly @16 @Float "y" n
  rows <- table @16 @Float "rows" n 64
  threads @32
  groups n
  pure . pure $ do
    acc <- var "acc" 0
    strided "i" 64 $ \i -> acc .= get acc + load x (groupIndex * 64 + i) * loadVia rows groupIndex i
    total <- simdSum "total" (get acc)
    whenLane (threadIndex .== 0) $ store y groupIndex total

-- | C = A · Bᵀ in 8×8 fragments, A and B 16 rows of k, B's tiles loaded transposed.
gemm :: forall a. (IsScalar a, AlignOk a 16, FragInput a) => Kernel AppleGPU9
gemm = kernel "gemm" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @a "a" (16 * k)
  b <- buffer @ReadOnly @16 @a "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure . pure $ do
    row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
    col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
    acc <- fragment @Float "acc"
    fa <- fragment @a "fa"
    fb <- fragment @a "fb"
    forU "p" 0 (k `quotE` 8) $ \p -> do
      fragLoad fa a (row0 * k + p * 8) k
      fragLoadT fb b (col0 * k + p * 8) k
      fragMma acc fa fb acc
    fragStore c (row0 * 16 + col0) 16 acc

{- | 'gemm' in bf16, A and B the arrays of entries 1 and 2 of a table, loaded
through it, and C entry 1's of another, stored through it.
-}
gemmVia :: Kernel AppleGPU9
gemmVia = kernel "gemm_via" $ do
  k <- dim "k" (DimRange 8 256 8)
  ab <- table @16 @BF16 "ab" 3 (16 * k)
  cs <- tableOf @WriteOnly @16 @Float "cs" 2 256
  distinct cs
  threads @128
  groups 1
  pure . pure $ do
    row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
    col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
    acc <- fragment @Float "acc"
    fa <- fragment @BF16 "fa"
    fb <- fragment @BF16 "fb"
    forU "p" 0 (k `quotE` 8) $ \p -> do
      fragLoadVia fa ab 1 (row0 * k + p * 8) k
      fragLoadViaT fb ab 2 (col0 * k + p * 8) k
      fragMma acc fa fb acc
    fragStoreVia cs 1 (row0 * 16 + col0) 16 acc

-- | 'gemm' in bf16 with B's tiles staged in threadgroup memory and loaded from there transposed.
staged :: Kernel AppleGPU9
staged = kernel "staged" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @BF16 "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure $ do
    tile <- tgArray @128 @BF16 "tile"
    pure $ do
      row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
      col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
      acc <- fragment @Float "acc"
      fa <- fragment @BF16 "fa"
      fb <- fragment @BF16 "fb"
      forU "p" 0 (k `quotE` 8) $ \p -> do
        tgStore tile threadIndex (load b ((threadIndex `quotE` 8) * k + p * 8 + threadIndex `remE` 8))
        barrier
        fragLoad fa a (row0 * k + p * 8) k
        fragLoadTgT fb tile (col0 * 8) 8
        fragMma acc fa fb acc
        barrier
      fragStore c (row0 * 16 + col0) 16 acc

-- | f32 A against bf16 B widened into f32 fragments, the product's rows scaled.
scaledRows :: Kernel AppleGPU9
scaledRows = kernel "scaled_rows" $ do
  k <- dim "k" (DimRange 8 256 8)
  a <- buffer @ReadOnly @16 @Float "a" (16 * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (16 * k)
  c <- buffer @WriteOnly @16 @Float "c" 256
  threads @128
  groups 1
  pure $ do
    scales <- tgArray @8 @Float "scales"
    pure $ do
      forM_ [0 .. 7 :: Int] $ \r -> whenLane (threadIndex .== 0) (tgStore scales (fromIntegral r) (fromIntegral (2 ^ r :: Int) * 0.3))
      barrier
      row0 <- let_ "row0" ((simdGroupIndex `quotE` 2) * 8)
      col0 <- let_ "col0" ((simdGroupIndex `remE` 2) * 8)
      acc <- fragment @Float "acc"
      fa <- fragment @Float "fa"
      fbh <- fragment @BF16 "fbh"
      fb <- fragment @Float "fb"
      forU "p" 0 (k `quotE` 8) $ \p -> do
        fragLoad fa a (row0 * k + p * 8) k
        fragLoadT fbh b (col0 * k + p * 8) k
        fragConvert fb fbh
        fragMma acc fa fb acc
      fragScaleRows acc scales 0
      fragStore c (row0 * 16 + col0) 16 acc

-- | C = A · Bᵀ of bf16 rows through a GEMM tile, edges clamped.
tiled :: forall na nb nc. (TgArrayOk AppleGPU9 na Float, TgArrayOk AppleGPU9 nb Float, TgArrayOk AppleGPU9 nc Float, KnownNat na, KnownNat nb, KnownNat nc) => GemmTile -> Kernel AppleGPU9
tiled t = kernel "tiled" $ do
  m <- dim "m" (DimRange 1 256 1)
  n <- dim "n" (DimRange 8 256 8)
  k <- dim "k" (DimRange t.gtStep 256 t.gtStep)
  a <- buffer @ReadOnly @16 @BF16 "a" (m * k)
  b <- buffer @ReadOnly @16 @BF16 "b" (n * k)
  c <- buffer @WriteOnly @16 @Float "c" (m * n)
  threads @256
  let rows = fromIntegral t.gtRows
      cols = fromIntegral t.gtCols
  groups (((m + rows - 1) `quotE` rows) * ((n + cols - 1) `quotE` cols))
  pure $ do
    aSlice <- tgArray @na @Float "a_slice"
    bSlice <- tgArray @nb @Float "b_slice"
    cTile <- tgArray @nc @Float "c_tile"
    pure $ do
      tilesN <- let_ "tiles_n" ((n + cols - 1) `quotE` cols)
      row0 <- let_ "row0" ((groupIndex `quotE` tilesN) * rows)
      col0 <- let_ "col0" ((groupIndex `remE` tilesN) * cols)
      gemmTile
        "t"
        t
        aSlice
        bSlice
        cTile
        k
        (\r p -> pure [widen @Float (load a (minE (row0 + r) (m - 1) * k + p + fromIntegral x)) | x <- [0 .. 3 :: Int]])
        (\col p -> pure [widen @Float (load b (minE (col0 + col) (n - 1) * k + p + fromIntegral x)) | x <- [0 .. 7 :: Int]])
        (\r col v -> whenLane (row0 + r .< m .&& col0 + col .< n) (store c ((row0 + r) * n + col0 + col) v))

-- | Every f32 operation on pairs of elements.
arithF32 :: Kernel AppleGPU9
arithF32 = kernel "arith_f32" $ do
  n <- dim "n" (DimRange 1 100000 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  y <- buffer @ReadOnly @16 @Float "y" n
  oAdd <- buffer @WriteOnly @16 @Float "o_add" n
  oSub <- buffer @WriteOnly @16 @Float "o_sub" n
  oMul <- buffer @WriteOnly @16 @Float "o_mul" n
  oDiv <- buffer @WriteOnly @16 @Float "o_div" n
  oMin <- buffer @WriteOnly @16 @Float "o_min" n
  oMax <- buffer @WriteOnly @16 @Float "o_max" n
  oSqrt <- buffer @WriteOnly @16 @Float "o_sqrt" n
  oRsqrt <- buffer @WriteOnly @16 @Float "o_rsqrt" n
  oExp <- buffer @WriteOnly @16 @Float "o_exp" n
  oLog <- buffer @WriteOnly @16 @Float "o_log" n
  oTanh <- buffer @WriteOnly @16 @Float "o_tanh" n
  oAbs <- buffer @WriteOnly @16 @Float "o_abs" n
  oNeg <- buffer @WriteOnly @16 @Float "o_neg" n
  oCmp <- buffer @WriteOnly @16 @Word32 "o_cmp" n
  oBf <- buffer @WriteOnly @16 @BF16 "o_bf" n
  oH <- buffer @WriteOnly @16 @F16 "o_h" n
  oSel <- buffer @WriteOnly @16 @Float "o_sel" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load x i)
      b <- let_ "b" (load y i)
      store oAdd i (a + b)
      store oSub i (a - b)
      store oMul i (a * b)
      store oDiv i (a / b)
      store oMin i (minE a b)
      store oMax i (maxE a b)
      store oSqrt i (sqrtE a)
      store oRsqrt i (rsqrtE a)
      store oExp i (expE a)
      store oLog i (logE a)
      store oTanh i (tanhE a)
      store oAbs i (abs a)
      store oNeg i (negate a)
      store oCmp i (select (a .< b) 1 0 + select (a .<= b) 2 0 + select (a .> b) 4 0 + select (a .>= b) 8 0 + select (a .== b) 16 0 + select (a ./= b) 32 0)
      store oBf i (narrow @BF16 a)
      store oH i (narrow @F16 a)
      store oSel i (select (a .< b) a b)

-- | Every f16 operation on pairs of elements.
arithF16 :: Kernel AppleGPU9
arithF16 = kernel "arith_f16" $ do
  n <- dim "n" (DimRange 1 100000 1)
  x <- buffer @ReadOnly @16 @F16 "x" n
  y <- buffer @ReadOnly @16 @F16 "y" n
  oAdd <- buffer @WriteOnly @16 @F16 "o_add" n
  oSub <- buffer @WriteOnly @16 @F16 "o_sub" n
  oMul <- buffer @WriteOnly @16 @F16 "o_mul" n
  oDiv <- buffer @WriteOnly @16 @F16 "o_div" n
  oMin <- buffer @WriteOnly @16 @F16 "o_min" n
  oMax <- buffer @WriteOnly @16 @F16 "o_max" n
  oSqrt <- buffer @WriteOnly @16 @F16 "o_sqrt" n
  oRsqrt <- buffer @WriteOnly @16 @F16 "o_rsqrt" n
  oExp <- buffer @WriteOnly @16 @F16 "o_exp" n
  oLog <- buffer @WriteOnly @16 @F16 "o_log" n
  oTanh <- buffer @WriteOnly @16 @F16 "o_tanh" n
  oAbs <- buffer @WriteOnly @16 @F16 "o_abs" n
  oNeg <- buffer @WriteOnly @16 @F16 "o_neg" n
  oCmp <- buffer @WriteOnly @16 @Word32 "o_cmp" n
  oWide <- buffer @WriteOnly @16 @Float "o_wide" n
  oLit <- buffer @WriteOnly @16 @F16 "o_lit" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load x i)
      b <- let_ "b" (load y i)
      store oAdd i (a + b)
      store oSub i (a - b)
      store oMul i (a * b)
      store oDiv i (a / b)
      store oMin i (minE a b)
      store oMax i (maxE a b)
      store oSqrt i (sqrtE a)
      store oRsqrt i (rsqrtE a)
      store oExp i (expE a)
      store oLog i (logE a)
      store oTanh i (tanhE a)
      store oAbs i (abs a)
      store oNeg i (negate a)
      store oCmp i (select (a .< b) 1 0 + select (a .<= b) 2 0 + select (a .> b) 4 0 + select (a .>= b) 8 0 + select (a .== b) 16 0 + select (a ./= b) 32 0)
      store oWide i (widen @Float a)
      store oLit i (a * 0.1 + (-3))

-- | Every i32 and u32 operation on pairs of elements, divisors that would be errors replaced.
arithInt :: Kernel AppleGPU9
arithInt = kernel "arith_int" $ do
  n <- dim "n" (DimRange 1 100000 1)
  xi <- buffer @ReadOnly @16 @Int32 "xi" n
  yi <- buffer @ReadOnly @16 @Int32 "yi" n
  xu <- buffer @ReadOnly @16 @Word32 "xu" n
  yu <- buffer @ReadOnly @16 @Word32 "yu" n
  iAdd <- buffer @WriteOnly @16 @Int32 "i_add" n
  iSub <- buffer @WriteOnly @16 @Int32 "i_sub" n
  iMul <- buffer @WriteOnly @16 @Int32 "i_mul" n
  iNeg <- buffer @WriteOnly @16 @Int32 "i_neg" n
  iAbs <- buffer @WriteOnly @16 @Int32 "i_abs" n
  iMin <- buffer @WriteOnly @16 @Int32 "i_min" n
  iMax <- buffer @WriteOnly @16 @Int32 "i_max" n
  iDiv <- buffer @WriteOnly @16 @Int32 "i_div" n
  iRem <- buffer @WriteOnly @16 @Int32 "i_rem" n
  iCmp <- buffer @WriteOnly @16 @Word32 "i_cmp" n
  uAdd <- buffer @WriteOnly @16 @Word32 "u_add" n
  uSub <- buffer @WriteOnly @16 @Word32 "u_sub" n
  uMul <- buffer @WriteOnly @16 @Word32 "u_mul" n
  uNeg <- buffer @WriteOnly @16 @Word32 "u_neg" n
  uDiv <- buffer @WriteOnly @16 @Word32 "u_div" n
  uRem <- buffer @WriteOnly @16 @Word32 "u_rem" n
  uMin <- buffer @WriteOnly @16 @Word32 "u_min" n
  uCmp <- buffer @WriteOnly @16 @Word32 "u_cmp" n
  uTof <- buffer @WriteOnly @16 @Float "u_tof" n
  iLit <- buffer @WriteOnly @16 @Int32 "i_lit" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load xi i)
      b <- let_ "b" (load yi i)
      p <- let_ "p" (load xu i)
      q <- let_ "q" (load yu i)
      store iAdd i (a + b)
      store iSub i (a - b)
      store iMul i (a * b)
      store iNeg i (negate a)
      store iAbs i (abs a)
      store iMin i (minE a b)
      store iMax i (maxE a b)
      d <- let_ "d" (select (b .== 0 .|| (b .== (-1) .&& a .== (-2147483647 - 1))) 7 b)
      store iDiv i (a `quotE` d)
      store iRem i (a `remE` d)
      store iCmp i (select (a .< b) 1 0 + select (a .<= b) 2 0 + select (a .> b) 4 0 + select (a .>= b) 8 0 + select (a .== b) 16 0 + select (a ./= b) 32 0)
      store uAdd i (p + q)
      store uSub i (p - q)
      store uMul i (p * q)
      store uNeg i (negate p)
      e <- let_ "e" (select (q .== 0) 3 q)
      store uDiv i (p `quotE` e)
      store uRem i (p `remE` e)
      store uMin i (minE p q + maxE p q * 3)
      store uCmp i (select (p .< q) 1 0 + select (p .<= q) 2 0 + select (p .> q) 4 0 + select (p .>= q) 8 0 + select (p .== q) 16 0 + select (p ./= q) 32 0)
      store uTof i (toF32 p)
      negated <- let_ "negated" (negate (-2147483647))
      store iLit i ((a * (-2147483647 - 1)) + (-1) - abs (-5) + negated)

-- | Every conversion.
conv :: Kernel AppleGPU9
conv = kernel "conv" $ do
  n <- dim "n" (DimRange 1 100000 1)
  xb <- buffer @ReadOnly @16 @BF16 "xb" n
  xh <- buffer @ReadOnly @16 @F16 "xh" n
  xs <- buffer @ReadOnly @16 @Int8 "xs" n
  xc <- buffer @ReadOnly @16 @Word8 "xc" n
  oB <- buffer @WriteOnly @16 @Float "o_b" n
  oH <- buffer @WriteOnly @16 @Float "o_h" n
  oS <- buffer @WriteOnly @16 @Int32 "o_s" n
  oC <- buffer @WriteOnly @16 @Word32 "o_c" n
  oRb <- buffer @WriteOnly @16 @BF16 "o_rb" n
  oRh <- buffer @WriteOnly @16 @F16 "o_rh" n
  oPb <- buffer @WriteOnly @16 @BF16 "o_pb" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      store oB i (widen @Float (load xb i))
      store oH i (widen @Float (load xh i))
      store oS i (widen @Int32 (load xs i))
      store oC i (widen @Word32 (load xc i))
      store oRb i (narrow @BF16 (widen @Float (load xb i)))
      store oRh i (narrow @F16 (widen @Float (load xh i)))
      store oPb i (load xb i)

-- | Divergent loops, SIMD-uniform branches, reductions, threadgroup memory.
flow :: Kernel AppleGPU9
flow = kernel "flow" $ do
  n <- dim "n" (DimRange 1 4096 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  cnt <- buffer @ReadOnly @16 @Word32 "cnt" 288
  y <- buffer @WriteOnly @16 @Float "y" 288
  z <- buffer @WriteOnly @16 @Float "z" 288
  w <- buffer @WriteOnly @16 @Word32 "w" 288
  threads @96
  groups 3
  pure $ do
    tg <- tgArray @96 @Float "tg"
    pure $ do
      g <- let_ "g" (groupIndex * 96 + threadIndex)
      acc <- var "acc" 0
      hits <- var "hits" 0
      forStep "j" (simdLane `remE` 3) (load cnt g) 2 $ \j -> do
        acc .= get acc + load x (j `remE` n)
        whenLane (j .== 4) $ do
          acc .= get acc * 1.5
          hits .= get hits + 1
      s <- simdSum "s" (get acc)
      mx <- simdMax "mx" (get acc)
      hs <- simdSum "hs" (get hits)
      whenSimd (simdGroupIndex .== 1) $ do
        t <- simdSum "t" (get acc * 3)
        whenLane (simdLane .< 5) $ store z g t
      tgStore tg threadIndex (s + mx)
      barrier
      store y g (tgLoad tg ((threadIndex + 1) `remE` 96))
      store w g hs
      barrier
      total <- tgReduceSum "total" tg (get acc)
      whenLane (threadIndex .== 0) $ store z (groupIndex * 96 + 95) total

-- | SIMD-group sums where the order shows, in f32 and in f16.
simdSums :: Kernel AppleGPU9
simdSums = kernel "simd_sums" $ do
  x <- buffer @ReadOnly @16 @Float "x" 64
  h <- buffer @ReadOnly @16 @F16 "h" 64
  y <- buffer @WriteOnly @16 @Float "y" 64
  g <- buffer @WriteOnly @16 @F16 "g" 64
  threads @64
  groups 1
  pure . pure $ do
    s <- simdSum "s" (load x threadIndex)
    store y threadIndex s
    t <- simdSum "t" (load h threadIndex)
    store g threadIndex t

{- | A group of 2^24 (2048 in f16) and ones, where lane order rounds every
one away, and one of spread magnitudes; then groups of 2^24 at lane 0 and
ones at lanes 16 and 17 only, which neighbours first sums to 2^24 + 2 and
lane order, or lane 16 first, to 2^24.
-}
simdSumsCase :: Case
simdSumsCase = Case "simd_sums" (checked "simd_sums" simdSums) [run xs hs, run (order 16777216) (order 2048)] []
 where
  run as bs = Inputs [] [("x", map VF32 as), ("h", map (VF16 . narrowF16) bs), ("y", zeros 64), ("g", replicate 64 (VF16 (narrowF16 0)))] []
  xs = [if l == 0 then 16777216 else 1 | l <- [0 .. 31 :: Int]] <> [encodeFloat (1 + fromIntegral l) (l `mod` 41 - 20) * (if even l then 1 else -1) | l <- [0 .. 31 :: Int]]
  hs = [if l == 0 then 2048 else 1 | l <- [0 .. 31 :: Int]] <> [encodeFloat (1 + fromIntegral l) (l `mod` 15 - 8) * (if even l then 1 else -1) | l <- [0 .. 31 :: Int]]
  order big = concat (replicate 2 [if l == 0 then big else if l `elem` [16, 17] then 1 else 0 | l <- [0 .. 31 :: Int]])

-- | fma of each triple, in f32.
fmaF32 :: Kernel AppleGPU9
fmaF32 = kernel "fma_f32" $ do
  n <- dim "n" (DimRange 1 100000 1)
  a <- buffer @ReadOnly @16 @Float "a" n
  b <- buffer @ReadOnly @16 @Float "b" n
  c <- buffer @ReadOnly @16 @Float "c" n
  o <- buffer @WriteOnly @16 @Float "o_fma" n
  threads @64
  groups 1
  pure . pure $ strided "i" n $ \i -> store o i (fmaE (load a i) (load b i) (load c i))

-- | fma of each triple, in f16.
fmaF16 :: Kernel AppleGPU9
fmaF16 = kernel "fma_f16" $ do
  n <- dim "n" (DimRange 1 100000 1)
  a <- buffer @ReadOnly @16 @F16 "a" n
  b <- buffer @ReadOnly @16 @F16 "b" n
  c <- buffer @ReadOnly @16 @F16 "c" n
  o <- buffer @WriteOnly @16 @F16 "o_fma" n
  threads @64
  groups 1
  pure . pure $ strided "i" n $ \i -> store o i (fmaE (load a i) (load b i) (load c i))

-- | Each i32 of a and u32 of b narrowed both ways.
narrowings :: Kernel AppleGPU9
narrowings = kernel "narrowings" $ do
  a <- buffer @ReadOnly @16 @Int32 "a" 16
  b <- buffer @ReadOnly @16 @Word32 "b" 16
  ca <- buffer @WriteOnly @16 @Int8 "clamped_a" 16
  wa <- buffer @WriteOnly @16 @Int8 "wrapped_a" 16
  cb <- buffer @WriteOnly @16 @Word8 "clamped_b" 16
  wb <- buffer @WriteOnly @16 @Word8 "wrapped_b" 16
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .< 16) $ do
    store ca threadIndex (narrowClamped @Int8 (load a threadIndex))
    store wa threadIndex (narrowWrapped @Int8 (load a threadIndex))
    store cb threadIndex (narrowClamped @Word8 (load b threadIndex))
    store wb threadIndex (narrowWrapped @Word8 (load b threadIndex))

-- | Each narrowing's edges and values past them.
narrowingsCase :: Case
narrowingsCase =
  Case
    "narrowings"
    (checked "narrowings" narrowings)
    [Inputs [] [("a", map VI32 as), ("b", map VU32 bs), ("clamped_a", i8s), ("wrapped_a", i8s), ("clamped_b", u8s), ("wrapped_b", u8s)] []]
    []
 where
  as = [minBound, minBound + 1, -32768, -300, -129, -128, -127, -1, 0, 1, 126, 127, 128, 300, 65535, maxBound]
  bs = [0, 1, 127, 128, 254, 255, 256, 257, 300, 511, 512, 65535, 65536, 2147483648, maxBound - 1, maxBound]
  i8s = replicate 16 (VI8 0)
  u8s = replicate 16 (VU8 0)

-- | SIMD-group and threadgroup maxima over NaNs and signed zeros.
reductions :: Kernel AppleGPU9
reductions = kernel "reductions" $ do
  x <- buffer @ReadOnly @16 @Float "x" 256
  y <- buffer @WriteOnly @16 @Float "y" 256
  z <- buffer @WriteOnly @16 @Float "z" 4
  threads @64
  groups 4
  pure $ do
    scratch <- tgArray @2 @Float "scratch"
    pure $ do
      g <- let_ "g" (groupIndex * 64 + threadIndex)
      v <- let_ "v" (load x g)
      m <- simdMax "m" v
      store y g m
      t <- tgReduceMax "t" scratch v
      whenLane (threadIndex .== 0) $ store z groupIndex t

-- | Parameters of every type, decoded from the struct's bytes.
params :: Kernel AppleGPU9
params = kernel "params" $ do
  n <- dim "n" (DimRange 1 64 1)
  a <- param @Bool "a"
  b <- param @Float "b"
  c <- param @F16 "c"
  d <- param @Int32 "d"
  e <- param @BF16 "e"
  f <- param @Word8 "f"
  g <- param @Int8 "g"
  y <- buffer @WriteOnly @16 @Float "y" n
  yh <- buffer @WriteOnly @16 @F16 "yh" n
  yi <- buffer @WriteOnly @16 @Int32 "yi" n
  threads @32
  groups 1
  pure . pure $ do
    whenLane (threadIndex .== 0) $ do
      store y 0 (select a b (widen @Float e))
      store y 1 (widen @Float e + toF32 (widen @Word32 f))
      store yh 0 (negate c * (-2.5) + 65504)
      store yi 0 (d + widen @Int32 g)

-- | Ones up to a dimension that is a multiple of 8 from 8 to 64.
ranged :: Kernel AppleGPU9
ranged = kernel "ranged" $ do
  n <- dim "n" (DimRange 8 64 8)
  y <- buffer @WriteOnly @16 @Float "y" 64
  threads @32
  groups 1
  pure . pure $ strided "i" n $ \i -> store y i 1

------------------------------------------------------------------------------
-- Inputs

-- | The outputs of arithmetic both arith cases have, whose NaNs' signs and payloads are the target's.
arithmetic :: [Name]
arithmetic = ["o_add", "o_sub", "o_mul", "o_div", "o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh"]

{- | Arithmetic's NaNs by class; minima's and maxima's whole, either zero,
and an operand as it was; the narrowings' NaNs whole; exp, log and tanh
within MSL's bounds; negation, magnitude, the comparisons and select to
their bits. Every output takes binary32's subnormal leave.
-}
arithF32Case :: Case
arithF32Case =
  Case "arith_f32" (checked "arith_f32" arithF32) [Inputs [("n", u count)] (inputs <> outputs) []] leaves
 where
  ps = pairs f32Specials <> [(fromIntegral (w .&. 0xFFFFFFFF), fromIntegral (w `shiftR` 32)) | w <- randoms 50 2000]
  count = length ps
  inputs = [("x", [VF32 (castWord32ToFloat a) | (a, _) <- ps]), ("y", [VF32 (castWord32ToFloat b) | (_, b) <- ps])]
  outputs =
    [(o, zeros count) | o <- ["o_add", "o_sub", "o_mul", "o_div", "o_min", "o_max", "o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh", "o_abs", "o_neg", "o_sel"]]
      <> [("o_cmp", replicate count (VU32 0)), ("o_bf", bf16Zeros count), ("o_h", replicate count (VF16 (F16 0)))]
  leaves =
    [ (arithmetic, [NanClass])
    , (["o_min", "o_max"], [AnyNan, Zeros, Operand])
    , (["o_bf", "o_h"], [AnyNan])
    , (["o_sel"], [Operand])
    , (["o_exp", "o_log"], [Ulp 4])
    , (["o_tanh"], [Ulp 5])
    , (map fst outputs, [Ftz])
    ]

{- | As 'arithF32Case', binary16 math rounded once from binary32 within 1
ulp, and no subnormal leave: binary16 has none.
-}
arithF16Case :: Case
arithF16Case =
  Case "arith_f16" (checked "arith_f16" arithF16) [Inputs [("n", u count)] (inputs <> outputs) []] leaves
 where
  ps = pairs f16Specials <> [(fromIntegral (w .&. 0xFFFF), fromIntegral (w `shiftR` 16 .&. 0xFFFF)) | w <- randoms 51 3000]
  count = length ps
  inputs = [("x", [VF16 (F16 a) | (a, _) <- ps]), ("y", [VF16 (F16 b) | (_, b) <- ps])]
  outputs =
    [(o, replicate count (VF16 (F16 0))) | o <- ["o_add", "o_sub", "o_mul", "o_div", "o_min", "o_max", "o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh", "o_abs", "o_neg", "o_lit"]]
      <> [("o_cmp", replicate count (VU32 0)), ("o_wide", zeros count)]
  leaves =
    [ ("o_lit" : arithmetic, [NanClass])
    , (["o_min", "o_max"], [AnyNan, Zeros])
    , (["o_wide"], [AnyNan])
    , (["o_rsqrt", "o_exp", "o_log", "o_tanh"], [Ulp 1])
    ]

arithIntCase :: Case
arithIntCase =
  Case "arith_int" (checked "arith_int" arithInt) [Inputs [("n", u count)] (inputs <> outputs) []] []
 where
  is = pairs i32Specials <> [(fromIntegral w, fromIntegral (w `shiftR` 32)) | w <- randoms 52 500]
  us = take count (cycle (pairs u32Specials <> [(fromIntegral w, fromIntegral (w `shiftR` 40)) | w <- randoms 53 500]))
  count = length is
  inputs = [("xi", [VI32 a | (a, _) <- is]), ("yi", [VI32 b | (_, b) <- is]), ("xu", [VU32 a | (a, _) <- us]), ("yu", [VU32 b | (_, b) <- us])]
  outputs =
    [(o, replicate count (VI32 0)) | o <- ["i_add", "i_sub", "i_mul", "i_neg", "i_abs", "i_min", "i_max", "i_div", "i_rem", "i_lit"]]
      <> [(o, replicate count (VU32 0)) | o <- ["i_cmp", "u_add", "u_sub", "u_mul", "u_neg", "u_div", "u_rem", "u_min", "u_cmp"]]
      <> [("u_tof", zeros count)]

{- | Specials against specials, over addends that make ties, cancellation
and overflow; random triples; each random product's rounding error, the
product plus its negated rounded value; and ties settled by an addend
below them.
-}
fmaF32Case :: Case
fmaF32Case = Case "fma_f32" (checked "fma_f32" fmaF32) [Inputs [("n", u count)] (inputs <> [("o_fma", zeros count)]) []] [(["o_fma"], [NanClass, Ftz])]
 where
  addends = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x00000001, 0x80000001, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x28800000, 0x33800000]
  edge = [(a, b, c) | a <- f32Specials, b <- f32Specials, c <- addends]
  rand = [(fromIntegral (w .&. 0xFFFFFFFF), fromIntegral (w `shiftR` 32), fromIntegral (v .&. 0xFFFFFFFF)) | (w, v) <- zip (randoms 56 3000) (randoms 57 3000)]
  errors = [(bits x, bits y, bits (negate (x * y))) | (x, y) <- zip (floats 58 3000) (floats 59 3000)]
  ties =
    [ (0x3F800001, 0x3F7FFFFF, 0x28800000)
    , (0x3F800001, 0x3F7FFFFF, 0xA8800000)
    , (0x3F800001, 0x3F800001, 0xBF800002)
    , (0x3F800800, 0x3F800800, 0xBF801000)
    , (0x7F7FFFFF, 0x40000000, 0xFF7FFFFF)
    , (0x1A000000, 0x1A400000, 0x00000000)
    , (0x9A000000, 0x19800000, 0x00000000)
    , (0x1A000000, 0x1A000000, 0x00000000)
    ]
  triples = edge <> rand <> errors <> ties
  count = length triples
  bits = castFloatToWord32
  asF32 = VF32 . castWord32ToFloat
  inputs = [("a", [asF32 a | (a, _, _) <- triples]), ("b", [asF32 b | (_, b, _) <- triples]), ("c", [asF32 c | (_, _, c) <- triples])]

{- | As 'fmaF32Case', in f16: through binary64 on the CPU target, exactly
in the interpreter, so every product's rounding error, every tie and every
subnormal shows where the two would differ.
-}
fmaF16Case :: Case
fmaF16Case = Case "fma_f16" (checked "fma_f16" fmaF16) [Inputs [("n", u count)] (inputs <> [("o_fma", replicate count (VF16 (F16 0)))]) []] [(["o_fma"], [NanClass])]
 where
  addends = [0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x1400, 0x3C02]
  edge = [(a, b, c) | a <- f16Specials, b <- f16Specials, c <- addends]
  rand = [(fromIntegral (w .&. 0xFFFF), fromIntegral (w `shiftR` 16 .&. 0xFFFF), fromIntegral (w `shiftR` 32 .&. 0xFFFF)) | w <- randoms 60 6000]
  errors = [(h x, h y, h (negate (widenF16 (narrowF16 (widenF16 (narrowF16 x) * widenF16 (narrowF16 y)))))) | (x, y) <- zip (floats 61 3000) (floats 62 3000)]
  ties = [(0x2740, 0x246A, 0x3C00), (0x2740, 0x246A, 0xBC00), (0x5BF8, 0x5C04, 0x0000), (0x5BF8, 0x5C04, 0xCC00), (0x3C01, 0x3C01, 0xBC02), (0x0C00, 0x0800, 0x0000), (0x0C00, 0x0A00, 0x0000), (0x8C00, 0x0400, 0x0000)]
  triples = edge <> rand <> errors <> ties
  count = length triples
  h v = let F16 b = narrowF16 v in b
  inputs = [(n, [VF16 (F16 (pick t)) | t <- triples]) | (n, pick) <- [("a", \(a, _, _) -> a), ("b", \(_, b, _) -> b), ("c", \(_, _, c) -> c)]]

{- | The conversions' NaNs whole. The narrowing of a widened bfloat16 may
take its binary32 subnormal as a zero; the widenings keep subnormals, and
the copy (o_pb) and the integers keep their bits.
-}
convCase :: Case
convCase =
  Case
    "conv"
    (checked "conv" conv)
    [ Inputs
        [("n", u count)]
        [ ("xb", bs)
        , ("xh", take count (cycle ([VF16 (F16 h) | h <- f16Specials] <> [VF16 (F16 (fromIntegral (w .&. 0xFFFF))) | w <- randoms 55 3000])))
        , ("xs", take count (cycle [VI8 (fromIntegral i) | i <- [0 .. 255 :: Int]]))
        , ("xc", take count (cycle [VU8 (fromIntegral i) | i <- [0 .. 255 :: Int]]))
        , ("o_b", zeros count)
        , ("o_h", zeros count)
        , ("o_s", replicate count (VI32 0))
        , ("o_c", replicate count (VU32 0))
        , ("o_rb", bf16Zeros count)
        , ("o_rh", replicate count (VF16 (F16 0)))
        , ("o_pb", bf16Zeros count)
        ]
        []
    ]
    [(["o_b", "o_h", "o_rb", "o_rh"], [AnyNan]), (["o_rb"], [Ftz])]
 where
  bs = [VBF16 (BF16 (fromIntegral (w .&. 0xFFFF))) | w <- randoms 54 4000] <> [VBF16 (BF16 b) | b <- [0x7F81, 0xFF81, 0x7FC0, 0x7FFF, 0x7F80, 0x0001, 0x8000, 0x0000, 0xFFC1]]
  count = length bs

{- | Threadgroups of NaNs alone, NaNs among numbers, signed zeros alone,
and numbers; maxima, so their NaNs whole and either zero.
-}
reductionsCase :: Case
reductionsCase = Case "reductions" (checked "reductions" reductions) [Inputs [] [("x", map VF32 xs), ("y", zeros 256), ("z", zeros 4)] []] [(["y", "z"], [AnyNan, Zeros])]
 where
  nans = map castWord32ToFloat [0x7FC00000, 0xFFC00001, 0x7F800001, 0xFF812345]
  zeroes = map castWord32ToFloat [0x00000000, 0x80000000]
  xs =
    take 64 (cycle nans)
      <> [if even i then nans !! (i `mod` 4) else fromIntegral (i - 32) | i <- [0 .. 63 :: Int]]
      <> [zeroes !! ((i `div` 3) `mod` 2) | i <- [0 .. 63 :: Int]]
      <> floats 62 63
      <> [castWord32ToFloat 0x7F800000]

rangedCase :: Case
rangedCase = Case "ranged" (checked "ranged" ranged) [Inputs [("n", u n)] [("y", zeros 64)] [] | n <- [16, 8, 64]] []

paramsCase :: Case
paramsCase =
  Case
    "params"
    (checked "params" params)
    [ Inputs [("n", u 4), ("a", VBool False), ("b", VF32 1.5), ("c", VF16 (narrowF16 0.75)), ("d", VI32 (-123456)), ("e", VBF16 (narrowBF16 (-2.25))), ("f", VU8 200), ("g", VI8 (-77))] outs []
    , Inputs [("n", u 4), ("a", VBool True), ("b", VF32 1.5), ("c", VF16 (narrowF16 (-1000))), ("d", VI32 maxBound), ("e", VBF16 (narrowBF16 3)), ("f", VU8 1), ("g", VI8 minBound)] outs []
    ]
    []
 where
  outs = [("y", zeros 4), ("yh", replicate 4 (VF16 (F16 0))), ("yi", replicate 4 (VI32 0))]

f32Specials :: [Word32]
f32Specials =
  [ 0x00000000
  , 0x80000000
  , 0x3F800000
  , 0xBF800000
  , 0x3F000000
  , 0x40000000
  , 0x40400000
  , 0x00000001
  , 0x007FFFFF
  , 0x00800000
  , 0x7F7FFFFF
  , 0x7F800000
  , 0xFF800000
  , 0x7FC00000
  , 0x7FC00001
  , 0xFFC00000
  , 0x7F800001
  , 0x7FA00000
  , 0xFF800001
  , 0x477FE000
  , 0x477FF000
  , 0x477FEFFF
  , 0x387FC000
  , 0x38800000
  , 0x33800000
  , 0x33000000
  , 0x33000001
  , 0x3F808000
  , 0x3F818000
  , 0x3F808001
  , 0x7F7F8000
  , 0x3EAAAAAB
  , 0x40490FDB
  , 0xC2C80000
  , 0x42C80000
  , 0x3F801000
  , 0x3F803000
  , 0xC1200000
  , 0x41200000
  , 0x3DCCCCCD
  ]

f16Specials :: [Word16]
f16Specials =
  [ 0x0000
  , 0x8000
  , 0x3C00
  , 0xBC00
  , 0x3800
  , 0x4000
  , 0x4200
  , 0x0001
  , 0x03FF
  , 0x0400
  , 0x7BFF
  , 0xFBFF
  , 0x7C00
  , 0xFC00
  , 0x7E00
  , 0x7E01
  , 0xFE00
  , 0x7C01
  , 0x7D00
  , 0x3555
  , 0x4248
  , 0x5640
  , 0x1400
  , 0x0200
  , 0x3C01
  , 0x7800
  , 0x5BFF
  , 0xC900
  ]

i32Specials :: [Int32]
i32Specials = [0, 1, -1, 2, -2, minBound, maxBound, 7, -7, 3, 65536, -65536, 46341, -46341, 1000000007]

u32Specials :: [Word32]
u32Specials =
  [ 0
  , 1
  , 2
  , 3
  , 0xFFFFFFFF
  , 0x80000000
  , 0x7FFFFFFF
  , 65536
  , 16777216
  , 16777217
  , 16777218
  , 16777219
  , 33554433
  , 33554434
  , 33554435
  , 33554438
  , 0xFFFFFF80
  , 0xFFFFFF7F
  , 0xFFFFFF81
  , 0x7FFFFFC0
  , 0x7FFFFFBF
  ]

pairs :: [a] -> [(a, a)]
pairs xs = [(a, b) | a <- xs, b <- xs]

-- | SplitMix's finaliser, over a seed's sequence.
randoms :: Word64 -> Int -> [Word64]
randoms seed n = [mix (seed * 0x9E3779B97F4A7C15 + fromIntegral i) | i <- [1 .. n]]
 where
  mix z0 =
    let z1 = (z0 `xor` (z0 `shiftR` 30)) * 0xBF58476D1CE4E5B9
        z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94D049BB133111EB
     in z2 `xor` (z2 `shiftR` 31)

-- | Values in [-2, 2), multiples of 2^-22.
floats :: Word64 -> Int -> [Float]
floats seed n = [fromIntegral (fromIntegral (w `shiftR` 40) - 2 ^ (23 :: Int) :: Int) / 2 ^ (22 :: Int) | w <- randoms seed n]

f32s, bf16s, f16s :: Word64 -> Int -> [Value]
f32s seed = map VF32 . floats seed
bf16s seed = map (VBF16 . narrowBF16) . floats seed
f16s seed = map (VF16 . narrowF16) . floats seed

i32v, u8v, i8v :: Word64 -> Int -> [Value]
i32v seed n = [VI32 (fromIntegral w) | w <- randoms seed n]
u8v seed n = [VU8 (fromIntegral w) | w <- randoms seed n]
i8v seed n = [VI8 (fromIntegral w) | w <- randoms seed n]

zeros, bf16Zeros :: Int -> [Value]
zeros n = replicate n (VF32 0)
bf16Zeros n = replicate n (VBF16 (narrowBF16 0))

u :: Int -> Value
u = VU32 . fromIntegral

------------------------------------------------------------------------------
-- This library's constructs

-- | Packs of two 16-bit elements, a ushort2 each: bf16 reversed, f16 as they are.
halves :: Kernel AppleGPU9
halves = kernel "halves" $ do
  a <- buffer @ReadOnly @16 @BF16 "a" 64
  b <- buffer @WriteOnly @16 @BF16 "b" 64
  c <- buffer @ReadOnly @16 @F16 "c" 64
  d <- buffer @WriteOnly @16 @F16 "d" 64
  threads @32
  groups 1
  pure . pure $ do
    p <- loadPack @2 "p" a threadIndex
    storePack @2 b threadIndex (reverse (unpack p))
    q <- loadPack @2 "q" c threadIndex
    storePack @2 d threadIndex (unpack q)

-- | Copies through packs of other widths: two f32, sixteen and two u8.
copies :: Kernel AppleGPU9
copies = kernel "copies" $ do
  a <- buffer @ReadOnly @16 @Float "a" 64
  b <- buffer @ReadWrite @16 @Float "b" 64
  c <- buffer @ReadOnly @16 @Word8 "c" 512
  d <- buffer @WriteOnly @16 @Word8 "d" 512
  e <- buffer @WriteOnly @16 @Word8 "e" 64
  threads @32
  groups 1
  pure . pure $ do
    f <- loadPack @2 "f" a threadIndex
    storePack @2 b threadIndex (unpack f)
    v <- loadPack @16 "u" c threadIndex
    storePack @16 d threadIndex (unpack v)
    h <- loadPack @2 "h" c threadIndex
    storePack @2 e threadIndex (reverse (unpack h))

{- | Four elements a thread stores to threadgroup memory at once, read back
after the barriers. The threadgroup barrier alone orders them; the
SIMD-group barrier before it, as every one, only the golden tests pin
('simdExchange').
-}
tgPacked :: Kernel AppleGPU9
tgPacked = kernel "tg_packed" $ do
  a <- buffer @ReadOnly @16 @Float "a" 1024
  b <- buffer @WriteOnly @16 @Float "b" 256
  threads @256
  groups 1
  pure $ do
    packed <- tgArray @1024 @Float "staged"
    plain <- tgArray @256 @Float "plain"
    pure $ do
      tgStorePack packed threadIndex [load a (threadIndex * 4 + fromIntegral e) | e <- [0 .. 3 :: Int]]
      tgStore plain threadIndex 0
      simdBarrier
      barrier
      store b threadIndex (tgLoad packed threadIndex + tgLoad plain threadIndex)

-- | A bf16 fragment widened into an f32 one: each element where it was.
widenedBf16 :: Kernel AppleGPU9
widenedBf16 = kernel "widened_bf16" $ do
  b <- buffer @ReadOnly @16 @BF16 "b" 64
  c <- buffer @WriteOnly @16 @Float "c" 64
  threads @32
  groups 1
  pure . pure $ do
    h <- fragment @BF16 "h"
    f <- fragment @Float "f"
    fragLoad h b 0 8
    fragConvert f h
    fragStore c 0 8 f

-- | An f16 fragment widened into an f32 one: each element where it was.
widenedF16 :: Kernel AppleGPU9
widenedF16 = kernel "widened_f16" $ do
  b <- buffer @ReadOnly @16 @F16 "b" 64
  c <- buffer @WriteOnly @16 @Float "c" 64
  threads @32
  groups 1
  pure . pure $ do
    h <- fragment @F16 "h"
    f <- fragment @Float "f"
    fragLoad h b 0 8
    fragConvert f h
    fragStore c 0 8 f

{- | Magnitudes of results Metal's compiler takes for non-negative (a
reciprocal square root, a square root, a square) and of a quotient, which
isn't; and the first magnitude negated.
-}
absResults :: Kernel AppleGPU9
absResults = kernel "abs_results" $ do
  n <- dim "n" (DimRange 1 100000 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  oRsqrt <- buffer @WriteOnly @16 @Float "o_rsqrt" n
  oSqrt <- buffer @WriteOnly @16 @Float "o_sqrt" n
  oSquare <- buffer @WriteOnly @16 @Float "o_square" n
  oQuot <- buffer @WriteOnly @16 @Float "o_quot" n
  oNeg <- buffer @WriteOnly @16 @Float "o_neg" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      a <- let_ "a" (load x i)
      store oRsqrt i (abs (rsqrtE a))
      store oSqrt i (abs (sqrtE a))
      store oSquare i (abs (a * a))
      store oQuot i (abs (1 / a))
      store oNeg i (negate (abs (rsqrtE a)))

-- | A multiply and an add written apart, which the source keeps unfused.
contraction :: Kernel AppleGPU9
contraction = kernel "contraction" $ do
  x <- buffer @ReadOnly @16 @Float "x" 3
  y <- buffer @WriteOnly @16 @Float "y" 2
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ do
    store y 0 (load x 0 * load x 1 + load x 2)
    store y 1 (fmaE (load x 0) (load x 1) (load x 2))

metalCases :: [Case]
metalCases =
  [ Case "halves" (checked "halves" halves) [Inputs [] [("a", bf16s 80 64), ("b", bf16Zeros 64), ("c", f16s 81 64), ("d", replicate 64 (VF16 (F16 0)))] []] []
  , Case "copies" (checked "copies" copies) [Inputs [] [("a", f32s 82 64), ("b", zeros 64), ("c", [VU8 (fromIntegral w) | w <- randoms 83 512]), ("d", replicate 512 (VU8 0)), ("e", replicate 64 (VU8 0))] []] []
  , Case "tg_packed" (checked "tg_packed" tgPacked) [Inputs [] [("a", f32s 84 1024), ("b", zeros 256)] []] []
  , Case "widened_bf16" (checked "widened_bf16" widenedBf16) [Inputs [] [("b", bf16s 85 64), ("c", zeros 64)] []] []
  , Case "widened_f16" (checked "widened_f16" widenedF16) [Inputs [] [("b", f16s 86 64), ("c", zeros 64)] []] []
  , -- a = b = 1 + 2^-12, c = -(1 + 2^-11): 0 unfused, 2^-24 fused.
    Case "contraction" (checked "contraction" contraction) [Inputs [] [("x", map VF32 [1 + 2 ** (-12), 1 + 2 ** (-12), -(1 + 2 ** (-11))]), ("y", zeros 2)] []] []
  , -- Item 14: abs(rsqrt(-0)) came out -inf through metal::abs. The
    -- specials, negative subnormals, NaNs of both signs, and spread values;
    -- arithmetic's NaNs by class, binary32's subnormal leave.
    Case
      "abs_results"
      (checked "abs_results" absResults)
      [Inputs [("n", u (length xs))] (("x", map (VF32 . castWord32ToFloat) xs) : [(o, zeros (length xs)) | o <- outs]) []]
      [(outs, [NanClass, Ftz])]
  ]
 where
  xs = f32Specials <> [0x80000001, 0x807FFFFF, 0x80400000, 0xFFC00001, 0xFF800002] <> [castFloatToWord32 v | v <- floats 103 200]
  outs = ["o_rsqrt", "o_sqrt", "o_square", "o_quot", "o_neg"]

------------------------------------------------------------------------------
-- Packs against element accesses

{- | A pack against element accesses of other memory: loaded and its
elements stored one at a time, and elements loaded one at a time and
stored as a pack, so that where a pack's elements sit in memory shows.
-}
mixedK :: forall a n. (IsScalar a, KnownNat n, PackOk AppleGPU9 n a 16, AlignOk a 16) => Name -> Kernel AppleGPU9
mixedK name = kernel name $ do
  let w = fromIntegral (natVal (Proxy @n)) :: Word32
      we = fromIntegral w :: Exp Word32
  x <- buffer @ReadOnly @16 @a "x" (32 * we)
  y <- buffer @WriteOnly @16 @a "y" (32 * we)
  z <- buffer @WriteOnly @16 @a "z" (32 * we)
  threads @32
  groups 1
  pure . pure $ do
    p <- loadPack @n "p" x threadIndex
    forM_ (zip [0 :: Int ..] (unpack p)) $ \(i, e) -> store y (threadIndex * we + fromIntegral i) e
    storePack @n z threadIndex [load x (threadIndex * we + fromIntegral i) | i <- [0 .. w - 1]]

{- | A pack loaded, an element stored, the pack loaded again; an element
loaded, a pack stored, the element loaded again: all in one thread, of its
own elements, which the compiler mustn't reorder.
-}
aliasK :: forall a n. (IsScalar a, KnownNat n, PackOk AppleGPU9 n a 16, AlignOk a 16) => Name -> Kernel AppleGPU9
aliasK name = kernel name $ do
  let w = fromIntegral (natVal (Proxy @n)) :: Word32
      we = fromIntegral w :: Exp Word32
  b <- buffer @ReadWrite @16 @a "b" (32 * we)
  c <- buffer @ReadWrite @16 @a "c" (32 * we)
  src <- buffer @ReadOnly @16 @a "src" 64
  o <- buffer @WriteOnly @16 @a "o" (64 * we)
  o2 <- buffer @WriteOnly @16 @a "o2" 64
  threads @32
  groups 1
  pure . pure $ do
    p <- loadPack @n "p" b threadIndex
    store b (threadIndex * we) (load src threadIndex)
    q <- loadPack @n "q" b threadIndex
    forM_ (zip [0 :: Int ..] (unpack p)) $ \(i, e) -> store o (threadIndex * (2 * we) + fromIntegral i) e
    forM_ (zip [0 :: Int ..] (unpack q)) $ \(i, e) -> store o (threadIndex * (2 * we) + we + fromIntegral i) e
    e1 <- let_ "e1" (load c (threadIndex * we))
    storePack @n c threadIndex [load src (32 + threadIndex) | _ <- [1 .. w]]
    e2 <- let_ "e2" (load c (threadIndex * we))
    store o2 (threadIndex * 2) e1
    store o2 (threadIndex * 2 + 1) e2

-- | As 'aliasK' in threadgroup memory: an element stored, read, a pack stored over it, read again.
tgAliasK :: forall a n. (IsScalar a, KnownNat n, AlignOk a 16, TgArrayOk AppleGPU9 256 a) => Name -> Kernel AppleGPU9
tgAliasK name = kernel name $ do
  let w = fromIntegral (natVal (Proxy @n)) :: Word32
      we = fromIntegral w :: Exp Word32
  src <- buffer @ReadOnly @16 @a "src" 64
  o2 <- buffer @WriteOnly @16 @a "o2" 96
  threads @32
  groups 1
  pure $ do
    t <- tgArray @256 @a "t"
    pure $ do
      tgStore t (threadIndex * we) (load src threadIndex)
      e1 <- let_ "e1" (tgLoad t (threadIndex * we))
      tgStorePack t threadIndex [load src (32 + threadIndex) | _ <- [1 .. w]]
      e2 <- let_ "e2" (tgLoad t (threadIndex * we))
      store o2 (threadIndex * 3) e1
      store o2 (threadIndex * 3 + 1) e2
      store o2 (threadIndex * 3 + 2) (tgLoad t (threadIndex * we + we - 1))

{- | A pack stored to threadgroup memory and read back an element at a time
by another lane, after the barrier: where a threadgroup pack's elements
sit shows.
-}
tgPackK :: forall a n. (IsScalar a, KnownNat n, AlignOk a 16, TgArrayOk AppleGPU9 512 a) => Name -> Kernel AppleGPU9
tgPackK name = kernel name $ do
  let w = fromIntegral (natVal (Proxy @n)) :: Word32
      we = fromIntegral w :: Exp Word32
  x <- buffer @ReadOnly @16 @a "x" (32 * we)
  y <- buffer @WriteOnly @16 @a "y" (32 * we)
  threads @32
  groups 1
  pure $ do
    t <- tgArray @512 @a "t"
    pure $ do
      tgStorePack t threadIndex [load x (threadIndex * we + fromIntegral i) | i <- [0 .. w - 1]]
      barrier
      forM_ [0 .. w - 1] $ \i -> store y (threadIndex * we + fromIntegral i) (tgLoad t ((31 - threadIndex) * we + fromIntegral i))

{- | Every container a pack goes through (ushort, ushort2, uint, uint2 and
uint4), over element types and widths: against element accesses of other
memory ('mixedK'), of its own ('aliasK', 'tgAliasK'), and stored to
threadgroup memory ('tgPackK'), as words and as vectors.
-}
packCases :: [Case]
packCases =
  [ mixed "mixed_u8x2" (mixedK @Word8 @2) 2 u8v
  , mixed "mixed_i8x2" (mixedK @Int8 @2) 2 i8v
  , mixed "mixed_bf16x2" (mixedK @BF16 @2) 2 bf16s
  , mixed "mixed_f16x2" (mixedK @F16 @2) 2 f16s
  , mixed "mixed_u8x4" (mixedK @Word8 @4) 4 u8v
  , mixed "mixed_f16x4" (mixedK @F16 @4) 4 f16s
  , mixed "mixed_f32x2" (mixedK @Float @2) 2 f32s
  , mixed "mixed_bf16x8" (mixedK @BF16 @8) 8 bf16s
  , mixed "mixed_f16x8" (mixedK @F16 @8) 8 f16s
  , mixed "mixed_f32x4" (mixedK @Float @4) 4 f32s
  , mixed "mixed_i32x4" (mixedK @Int32 @4) 4 i32v
  , mixed "mixed_u8x16" (mixedK @Word8 @16) 16 u8v
  , mixed "mixed_i8x16" (mixedK @Int8 @16) 16 i8v
  , alias "alias_c2" (aliasK @Int8 @2) 2 i8v
  , alias "alias_h2" (aliasK @F16 @2) 2 f16s
  , alias "alias_b2" (aliasK @BF16 @2) 2 bf16s
  , alias "alias_u4" (aliasK @Word8 @4) 4 u8v
  , alias "alias_h4" (aliasK @F16 @4) 4 f16s
  , alias "alias_f2" (aliasK @Float @2) 2 f32s
  , alias "alias_h8" (aliasK @F16 @8) 8 f16s
  , alias "alias_b8" (aliasK @BF16 @8) 8 bf16s
  , alias "alias_f4" (aliasK @Float @4) 4 f32s
  , alias "alias_i4" (aliasK @Int32 @4) 4 i32v
  , alias "alias_u16" (aliasK @Word8 @16) 16 u8v
  , alias "alias_c16" (aliasK @Int8 @16) 16 i8v
  , tgAlias "tgalias_h2" (tgAliasK @F16 @2) f16s
  , tgAlias "tgalias_h8" (tgAliasK @F16 @8) f16s
  , tgAlias "tgalias_b4" (tgAliasK @BF16 @4) bf16s
  , tgAlias "tgalias_f2" (tgAliasK @Float @2) f32s
  , tgAlias "tgalias_f4" (tgAliasK @Float @4) f32s
  , tgAlias "tgalias_i2" (tgAliasK @Int32 @2) i32v
  , tgAlias "tgalias_u8" (tgAliasK @Word8 @8) u8v
  , tgAlias "tgalias_c8" (tgAliasK @Int8 @8) i8v
  , tgPack "tgpack_f16x2" (tgPackK @F16 @2) 2 f16s
  , tgPack "tgpack_u8x4" (tgPackK @Word8 @4) 4 u8v
  , tgPack "tgpack_f32x2" (tgPackK @Float @2) 2 f32s
  , tgPack "tgpack_bf16x8" (tgPackK @BF16 @8) 8 bf16s
  , tgPack "tgpack_i32x4" (tgPackK @Int32 @4) 4 i32v
  ]
 where
  mixed name k w gen = Case name (checked name (k name)) [Inputs [] [("x", gen 90 (32 * w)), ("y", gen 91 (32 * w)), ("z", gen 92 (32 * w))] []] []
  alias name k w gen = Case name (checked name (k name)) [Inputs [] [("b", gen 1 (32 * w)), ("c", gen 2 (32 * w)), ("src", gen 3 64), ("o", gen 4 (64 * w)), ("o2", gen 5 64)] []] []
  tgAlias name k gen = Case name (checked name (k name)) [Inputs [] [("src", gen 3 64), ("o2", gen 5 96)] []] []
  tgPack name k w gen = Case name (checked name (k name)) [Inputs [] [("x", gen 93 (32 * w)), ("y", gen 94 (32 * w))] []] []

------------------------------------------------------------------------------
-- Coverage: constructs, limits and values the cases above leave out

{- | Lanes exchanging through threadgroup memory within their SIMD group,
ordered by its barrier alone: each reads another lane's store. On the M3
Max it passes with the barrier removed too, as every case does with every
SIMD-group barrier removed, the GPU's SIMD groups seeming to run in
lockstep: only the golden tests pin the barrier's lowering.
-}
simdExchange :: Kernel AppleGPU9
simdExchange = kernel "simd_exchange" $ do
  x <- buffer @ReadOnly @16 @Float "x" 128
  y <- buffer @WriteOnly @16 @Float "y" 128
  threads @128
  groups 1
  pure $ do
    t <- tgArray @128 @Float "t"
    pure $ do
      tgStore t (simdGroupIndex * 32 + simdLane) (load x (simdGroupIndex * 32 + simdLane))
      simdBarrier
      store y (simdGroupIndex * 32 + simdLane) (tgLoad t (simdGroupIndex * 32 + 31 - simdLane))

{- | Fragments through threadgroup memory, ordered by SIMD-group barriers:
one stored and read back by the lanes, reversed, and lanes' stores loaded
as one. As 'simdExchange' does, it passes on the M3 Max without them too.
-}
fragTg :: Kernel AppleGPU9
fragTg = kernel "frag_tg" $ do
  src <- buffer @ReadOnly @16 @Float "src" 256
  o <- buffer @WriteOnly @16 @Float "o" 256
  o2 <- buffer @WriteOnly @16 @Float "o2" 256
  threads @128
  groups 1
  pure $ do
    tg <- tgArray @256 @Float "tg"
    tg2 <- tgArray @256 @Float "tg2"
    pure $ do
      f <- fragment @Float "f"
      fragLoad f src (simdGroupIndex * 64) 8
      fragStoreTg tg (simdGroupIndex * 64) 8 f
      simdBarrier
      store o (simdGroupIndex * 64 + simdLane) (tgLoad tg (simdGroupIndex * 64 + 63 - simdLane))
      store o (simdGroupIndex * 64 + 32 + simdLane) (tgLoad tg (simdGroupIndex * 64 + 31 - simdLane))
      tgStore tg2 (simdGroupIndex * 64 + simdLane) (load src (255 - (simdGroupIndex * 64 + simdLane)))
      tgStore tg2 (simdGroupIndex * 64 + 32 + simdLane) (load src (255 - (simdGroupIndex * 64 + 32 + simdLane)))
      simdBarrier
      g <- fragment @Float "g"
      fragLoadTg g tg2 (simdGroupIndex * 64) 8
      fragStore o2 (simdGroupIndex * 64) 8 g

-- | Not and the bool literals, over comparisons of NaNs, infinities and signed zeros.
bools :: Kernel AppleGPU9
bools = kernel "bools" $ do
  n <- dim "n" (DimRange 1 64 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  y <- buffer @ReadOnly @16 @Float "y" n
  o <- buffer @WriteOnly @16 @Bool "o" n
  c <- buffer @WriteOnly @16 @Word32 "c" n
  threads @32
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      lt <- let_ "lt" (load x i .< load y i)
      store o i (notE lt)
      store c i (select (notE lt .&& true_) 1 0 + select (lt .|| false_) 2 0 + select (notE (notE lt)) 4 0 + select (notE false_) 8 0)

-- | Packs of two and four through a table.
packVia :: Kernel AppleGPU9
packVia = kernel "pack_via" $ do
  th <- table @16 @F16 "th" 2 64
  o <- buffer @WriteOnly @16 @F16 "o" 192
  threads @32
  groups 1
  pure . pure $ do
    p <- loadPackVia @2 "p" th 1 threadIndex
    q <- loadPackVia @4 "q" th 0 (threadIndex `remE` 16)
    forM_ (zip [0 :: Int ..] (unpack p <> unpack q)) $ \(i, e) -> store o (threadIndex * 6 + fromIntegral i) e

-- | Every element up to a dimension, run at its range's ends.
ends :: Kernel AppleGPU9
ends = kernel "ends" $ do
  n <- dim "n" (DimRange 1 4096 1)
  x <- buffer @ReadOnly @16 @Float "x" n
  y <- buffer @WriteOnly @16 @Float "y" n
  threads @64
  groups 1
  pure . pure $ strided "i" n $ \i -> store y i (load x i * 2 + toF32 i)

-- | 1024 threads, and threadgroup memory of 32 KiB, every byte of it.
tgFull :: Kernel AppleGPU9
tgFull = kernel "tg_full" $ do
  o <- buffer @WriteOnly @16 @Float "o" 1024
  threads @1024
  groups 1
  pure $ do
    a <- tgArray @8188 @Float "a"
    b <- tgArray @3 @F16 "b"
    pure $ do
      strided "i" 8188 $ \i -> tgStore a i (toF32 i)
      whenLane (threadIndex .< 3) $ tgStore b threadIndex (narrow @F16 (toF32 threadIndex))
      barrier
      store o threadIndex (tgLoad a (8187 - threadIndex) + widen @Float (tgLoad b (threadIndex `remE` 3)))

-- | 31 bindings, AppleGPU9's most: the parameter struct, 29 buffers and a table.
manyBindings :: Kernel AppleGPU9
manyBindings = kernel "many_bindings" $ do
  s <- param @Float "s"
  bs <- forM [0 .. 28 :: Int] $ \i -> buffer @ReadWrite @16 @Float ("b" <> T.pack (show i)) 32
  tb <- table @16 @Float "tb" 1 32
  threads @32
  groups 1
  pure . pure $ forM_ bs $ \b -> store b threadIndex (load b threadIndex * s + loadVia tb 0 threadIndex)

-- | A parameter struct of 1020 binary32s, 4080 bytes: setBytes' 4 KB nearly.
bigParams :: Kernel AppleGPU9
bigParams = kernel "big_params" $ do
  ps <- forM [0 .. 1019 :: Int] $ \i -> param @Float ("p" <> T.pack (show i))
  y <- buffer @WriteOnly @16 @Float "y" 1020
  threads @32
  groups 1
  pure . pure $ whenLane (threadIndex .== 0) $ forM_ (zip [0 :: Int ..] ps) $ \(i, p) -> store y (fromIntegral i) p

-- | SIMD-group maxima over subnormal lanes, signed zeros among them.
simdMaxSub :: Kernel AppleGPU9
simdMaxSub = kernel "simd_max_sub" $ do
  x <- buffer @ReadOnly @16 @Float "x" 128
  y <- buffer @WriteOnly @16 @Float "y" 128
  threads @128
  groups 1
  pure . pure $ do
    m <- simdMax "m" (load x threadIndex)
    store y threadIndex m

{- | Every binary16 through the unary operations and a widening. (A select
of a comparison, compiled with Metal's default options, takes a NaN-ignoring
minimum: item 43.)
-}
f16Sweep :: Kernel AppleGPU9
f16Sweep = kernel "f16_sweep" $ do
  x <- buffer @ReadOnly @16 @F16 "x" 65536
  oSqrt <- buffer @WriteOnly @16 @F16 "o_sqrt" 65536
  oRsqrt <- buffer @WriteOnly @16 @F16 "o_rsqrt" 65536
  oExp <- buffer @WriteOnly @16 @F16 "o_exp" 65536
  oLog <- buffer @WriteOnly @16 @F16 "o_log" 65536
  oTanh <- buffer @WriteOnly @16 @F16 "o_tanh" 65536
  oAbs <- buffer @WriteOnly @16 @F16 "o_abs" 65536
  oNeg <- buffer @WriteOnly @16 @F16 "o_neg" 65536
  oWide <- buffer @WriteOnly @16 @Float "o_wide" 65536
  threads @256
  groups 256
  pure . pure $ do
    i <- let_ "i" (groupIndex * 256 + threadIndex)
    a <- let_ "a" (load x i)
    store oSqrt i (sqrtE a)
    store oRsqrt i (rsqrtE a)
    store oExp i (expE a)
    store oLog i (logE a)
    store oTanh i (tanhE a)
    store oAbs i (abs a)
    store oNeg i (negate a)
    store oWide i (widen @Float a)

-- | Every bfloat16, copied and widened.
bf16Sweep :: Kernel AppleGPU9
bf16Sweep = kernel "bf16_sweep" $ do
  x <- buffer @ReadOnly @16 @BF16 "x" 65536
  oCopy <- buffer @WriteOnly @16 @BF16 "o_copy" 65536
  oWide <- buffer @WriteOnly @16 @Float "o_wide" 65536
  threads @256
  groups 256
  pure . pure $ do
    i <- let_ "i" (groupIndex * 256 + threadIndex)
    store oCopy i (load x i)
    store oWide i (widen @Float (load x i))

-- | Every bfloat16 through packs of two and of eight, a ushort2's and a uint4's elements, widened.
bf16PackedSweep :: Kernel AppleGPU9
bf16PackedSweep = kernel "bf16_packed_sweep" $ do
  x <- buffer @ReadOnly @16 @BF16 "x" 65536
  o2 <- buffer @WriteOnly @16 @Float "o2" 65536
  o8 <- buffer @WriteOnly @16 @Float "o8" 65536
  threads @256
  groups 32
  pure . pure $ do
    i <- let_ "i" (groupIndex * 256 + threadIndex)
    e <- loadPack @8 "e" x i
    forM_ (zip [0 :: Int ..] (unpack e)) $ \(k, v) -> store o8 (i * 8 + fromIntegral k) (widen @Float v)
    forM_ [0 .. 3 :: Int] $ \j -> do
      t <- loadPack @2 (T.pack ("t" <> show j)) x (i * 4 + fromIntegral j)
      forM_ (zip [0 :: Int ..] (unpack t)) $ \(k, v) -> store o2 ((i * 4 + fromIntegral j) * 2 + fromIntegral k) (widen @Float v)

coverageCases :: [Case]
coverageCases =
  [ Case "simd_exchange" (checked "simd_exchange" simdExchange) [Inputs [] [("x", f32s 100 128), ("y", zeros 128)] []] []
  , Case "frag_tg" (checked "frag_tg" fragTg) [Inputs [] [("src", f32s 31 256), ("o", zeros 256), ("o2", zeros 256)] []] []
  , Case "bools" (checked "bools" bools) [Inputs [("n", u 64)] [("x", map VF32 xs), ("y", map VF32 ys), ("o", replicate 64 (VBool False)), ("c", replicate 64 (VU32 0))] []] []
  , Case "pack_via" (checked "pack_via" packVia) [Inputs [] [("o", f16s 51 192)] [("th", [f16s 52 64, f16s 53 64])]] []
  , Case "ends" (checked "ends" ends) [Inputs [("n", u n)] [("x", f32s 101 n), ("y", zeros n)] [] | n <- [1, 4096]] []
  , Case "tg_full" (checked "tg_full" tgFull) [Inputs [] [("o", zeros 1024)] []] []
  , Case "many_bindings" (checked "many_bindings" manyBindings) [Inputs [("s", VF32 1.5)] [("b" <> T.pack (show i), f32s (500 + fromIntegral i) 32) | i <- [0 .. 28 :: Int]] [("tb", [f32s 600 32])]] []
  , Case "big_params" (checked "big_params" bigParams) [Inputs [("p" <> T.pack (show i), v) | (i, v) <- zip [0 :: Int ..] (f32s 102 1020)] [("y", zeros 1020)] []] []
  , Case "simd_max_sub" (checked "simd_max_sub" simdMaxSub) [Inputs [] [("x", map (VF32 . castWord32ToFloat) maxima), ("y", zeros 128)] []] [(["y"], [AnyNan, Ftz, Operand, Zeros])]
  , gemmSpecialsCase
  , Case "f16_sweep" (checked "f16_sweep" f16Sweep) [Inputs [] (("x", [VF16 (F16 h) | h <- [0 .. 65535]]) : sweepOuts) []] sweepLeaves
  , Case "bf16_sweep" (checked "bf16_sweep" bf16Sweep) [Inputs [] [("x", [VBF16 (BF16 h) | h <- [0 .. 65535]]), ("o_copy", bf16Zeros 65536), ("o_wide", zeros 65536)] []] [(["o_wide"], [AnyNan])]
  , Case "bf16_packed_sweep" (checked "bf16_packed_sweep" bf16PackedSweep) [Inputs [] [("x", [VBF16 (BF16 h) | h <- [0 .. 65535]]), ("o2", zeros 65536), ("o8", zeros 65536)] []] [(["o2", "o8"], [AnyNan])]
  ]
 where
  specials = map castWord32ToFloat [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001]
  (xs, ys) = unzip [(a, b) | a <- specials, b <- specials]
  -- Lanes of subnormals of both signs; subnormals and signed zeros; signed
  -- zeros alone; negative subnormals and negative normals.
  sub k = (k * 0x3FFF + 1) .&. 0x007FFFFF :: Word32
  maxima =
    [(if even l then 0 else 0x80000000) .|. sub l | l <- [0 .. 31]]
      <> [if l `mod` 4 == 0 then 0 else if l `mod` 4 == 1 then 0x80000000 else (if even (l `div` 4) then 0 else 0x80000000) .|. sub (l + 40) | l <- [0 .. 31]]
      <> [if odd l then 0x80000000 else 0 | l <- [0 .. 31 :: Word32]]
      <> [if l < 16 then 0x80000000 .|. sub (l + 80) else 0x80800000 + l | l <- [0 .. 31]]
  sweepOuts = [(o, replicate 65536 (VF16 (F16 0))) | o <- ["o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh", "o_abs", "o_neg"]] <> [("o_wide", zeros 65536)]
  -- As arith_f16's: binary16 math rounded once from binary32, sqrt exactly
  -- and the others within 1 ulp; a widening's NaN the target's; magnitude
  -- and negation to their bits.
  sweepLeaves = [(["o_sqrt", "o_rsqrt", "o_exp", "o_log", "o_tanh"], [NanClass]), (["o_rsqrt", "o_exp", "o_log", "o_tanh"], [Ulp 1]), (["o_wide"], [AnyNan])]

{- | 'gemm' in f32 over NaNs, infinities and subnormals: rows and columns
where a NaN, an infinity of either sign, both, or an infinity times zero
decide the products, and rows of subnormals, alone and among normal
values, small enough that their products and sums stay subnormal.
-}
gemmSpecialsCase :: Case
gemmSpecialsCase =
  Case "gemm_specials" (checked "gemm_specials" (gemm @Float)) [Inputs [("k", u 64)] [("a", plant 97 aSpecials), ("b", plant 98 bSpecials), ("c", zeros 256)] []] [(["c"], [NanClass, Ftz])]
 where
  -- Elements by row (A) or column (B) and their index in K.
  plant seed specials = [maybe v (VF32 . castWord32ToFloat) (lookup (i `divMod` 64) specials) | (i, v) <- zip [0 :: Int ..] (f32s seed (16 * 64))]
  aSpecials =
    [((0, 3), 0x7FC00000), ((1, 10), 0x7F800001), ((6, 7), 0x00000000)]
      <> [((3, p), subnormal p) | p <- [0 .. 63]]
      <> [((4, p), subnormal p) | p <- [0, 2 .. 62]]
  bSpecials = [((1, 5), 0x7F800000), ((1, 9), 0xFF800000), ((2, 7), 0x7F800000), ((5, 20), 0x00000005)]
  subnormal p = (if even p then 0 else 0x80000000) .|. (fromIntegral p * 7 + 1)

------------------------------------------------------------------------------
-- Metal's miscompiles: each one's repro, which fails without its workaround

-- | Item 6: f16(max(65520, x)), min's and select's alike, rounded toward zero and saturated.
narrowMaxLiteral :: Kernel AppleGPU9
narrowMaxLiteral = kernel "narrow_max_literal" $ do
  x <- buffer @ReadOnly @16 @Float "x" 64
  y <- buffer @WriteOnly @16 @F16 "y" 64
  z <- buffer @WriteOnly @16 @F16 "z" 64
  w <- buffer @WriteOnly @16 @F16 "w" 64
  threads @64
  groups 1
  pure . pure $ do
    store y threadIndex (narrow @F16 (maxE 65520 (load x threadIndex)))
    store z threadIndex (narrow @F16 (minE (-65520) (load x threadIndex)))
    store w threadIndex (narrow @F16 (select (load x threadIndex .< 0) 65520 (load x threadIndex)))

-- | Item 1: a parameter narrowed to bf16 with another use, given f16's encoding.
paramToBf16 :: Kernel AppleGPU9
paramToBf16 = kernel "param_to_bf16" $ do
  s <- param @Float "s"
  y <- buffer @WriteOnly @16 @BF16 "y" 32
  z <- buffer @WriteOnly @16 @Float "z" 32
  threads @32
  groups 1
  pure . pure $ do
    store y threadIndex (narrow @BF16 s)
    store z threadIndex s

{- | Item 1's characterization in one kernel: narrowed to bf16, a load at
the group's index, a SIMD-group sum and a value per thread, which narrow
right natively, beside a parameter, used whole and narrowed to f16 too,
which doesn't.
-}
narrowMixed :: Kernel AppleGPU9
narrowMixed = kernel "narrow_mixed" $ do
  s <- param @Float "s"
  x <- buffer @ReadOnly @16 @Float "x" 128
  oG <- buffer @WriteOnly @16 @BF16 "o_g" 128
  oS <- buffer @WriteOnly @16 @BF16 "o_s" 128
  oT <- buffer @WriteOnly @16 @BF16 "o_t" 128
  oP <- buffer @WriteOnly @16 @BF16 "o_p" 128
  oH <- buffer @WriteOnly @16 @F16 "o_h" 128
  oF <- buffer @WriteOnly @16 @Float "o_f" 128
  threads @64
  groups 2
  pure . pure $ do
    i <- let_ "i" (groupIndex * 64 + threadIndex)
    total <- simdSum "total" (load x i)
    store oG i (narrow @BF16 (load x groupIndex))
    store oS i (narrow @BF16 total)
    store oT i (narrow @BF16 (load x i * s))
    store oP i (narrow @BF16 s)
    store oH i (narrow @F16 s)
    store oF i s

-- | IF-1's kernel: a parameter narrowed to bf16 and to f16 in a strided loop.
bothNarrowings :: Kernel AppleGPU9
bothNarrowings = kernel "both_narrowings" $ do
  n <- dim "n" (DimRange 1 64 1)
  s <- param @Float "s"
  yb <- buffer @WriteOnly @16 @BF16 "yb" n
  yh <- buffer @WriteOnly @16 @F16 "yh" n
  threads @64
  groups 1
  pure . pure $
    strided "i" n $ \i -> do
      store yb i (narrow @BF16 s)
      store yh i (narrow @F16 s)

-- | Item 1's controls: the parameter's one use, and one through a var and a select.
paramToBf16Once, paramToBf16Var :: Kernel AppleGPU9
paramToBf16Once = kernel "param_to_bf16_once" $ do
  s <- param @Float "s"
  y <- buffer @WriteOnly @16 @BF16 "y" 32
  threads @32
  groups 1
  pure . pure $ store y threadIndex (narrow @BF16 s)
paramToBf16Var = kernel "param_to_bf16_var" $ do
  s <- param @Float "s"
  x <- buffer @ReadOnly @16 @Float "x" 32
  y <- buffer @WriteOnly @16 @BF16 "y" 32
  z <- buffer @WriteOnly @16 @Float "z" 32
  threads @32
  groups 1
  pure . pure $ do
    h <- var "h" (narrow @BF16 (select false_ (load x threadIndex) s))
    store y threadIndex (get h)
    store z threadIndex (widen @Float (get h))

-- | Item 2: a bf16 select, through float: NaNs canonical, subnormals flushed.
selectBf16 :: Kernel AppleGPU9
selectBf16 = kernel "select_bf16" $ do
  a <- buffer @ReadOnly @16 @BF16 "a" 64
  c <- buffer @ReadOnly @16 @Bool "c" 32
  y <- buffer @WriteOnly @16 @BF16 "y" 32
  threads @32
  groups 1
  pure . pure $ store y threadIndex (select (load c threadIndex) (load a threadIndex) (load a (threadIndex + 32)))

-- | Item 2's control: f16's and f32's selects of the same shape.
selectF16F32 :: Kernel AppleGPU9
selectF16F32 = kernel "select_f16_f32" $ do
  a <- buffer @ReadOnly @16 @F16 "a" 64
  b <- buffer @ReadOnly @16 @Float "b" 64
  c <- buffer @ReadOnly @16 @Bool "c" 32
  y <- buffer @WriteOnly @16 @F16 "y" 32
  z <- buffer @WriteOnly @16 @Float "z" 32
  threads @32
  groups 1
  pure . pure $ do
    store y threadIndex (select (load c threadIndex) (load a threadIndex) (load a (threadIndex + 32)))
    store z threadIndex (select (load c threadIndex) (load b threadIndex) (load b (threadIndex + 32)))

-- | Item 5: u32 literal dividends at 2^32's edge over run-time divisors.
udivLiteral :: Kernel AppleGPU9
udivLiteral = kernel "udiv_literal" $ do
  d <- buffer @ReadOnly @16 @Word32 "d" 64
  q <- buffer @WriteOnly @16 @Word32 "q" 64
  r <- buffer @WriteOnly @16 @Word32 "r" 64
  threads @64
  groups 1
  pure . pure $ do
    store q threadIndex (4294967295 `quotE` load d threadIndex)
    store r threadIndex (4294967168 `remE` load d threadIndex)

-- | Item 3: every f16 over a constant divisor, 1 ulp off.
f16DivLiteral :: Kernel AppleGPU9
f16DivLiteral = kernel "f16_div_literal" $ do
  x <- buffer @ReadOnly @16 @F16 "x" 65536
  y <- buffer @WriteOnly @16 @F16 "y" 65536
  threads @256
  groups 256
  pure . pure $ do
    i <- let_ "i" (groupIndex * 256 + threadIndex)
    store y i (load x i / 3)

-- | Item 4: f32 sqrt and rsqrt of every f16 and every bf16, widened.
sqrtWidened :: Kernel AppleGPU9
sqrtWidened = kernel "sqrt_widened" $ do
  h <- buffer @ReadOnly @16 @F16 "h" 65536
  b <- buffer @ReadOnly @16 @BF16 "b" 65536
  hs <- buffer @WriteOnly @16 @Float "hs" 65536
  hr <- buffer @WriteOnly @16 @Float "hr" 65536
  bs <- buffer @WriteOnly @16 @Float "bs" 65536
  br <- buffer @WriteOnly @16 @Float "br" 65536
  threads @256
  groups 256
  pure . pure $ do
    i <- let_ "i" (groupIndex * 256 + threadIndex)
    store hs i (sqrtE (widen @Float (load h i)))
    store hr i (rsqrtE (widen @Float (load h i)))
    store bs i (sqrtE (widen @Float (load b i)))
    store br i (rsqrtE (widen @Float (load b i)))

-- | Item 43: every half's select of a comparison, an fmin under Metal's default options.
f16Select :: Kernel AppleGPU9
f16Select = kernel "f16_select" $ do
  x <- buffer @ReadOnly @16 @F16 "x" 65536
  y <- buffer @ReadOnly @16 @F16 "y" 65536
  o <- buffer @WriteOnly @16 @F16 "o" 65536
  threads @256
  groups 256
  pure . pure $ do
    i <- let_ "i" (groupIndex * 256 + threadIndex)
    a <- let_ "a" (load x i)
    b <- let_ "b" (load y i)
    store o i (select (a .< b) a b)

reproCases :: [Case]
reproCases =
  [ Case "narrow_max_literal" (checked "narrow_max_literal" narrowMaxLiteral) [Inputs [] [("x", map VF32 big), ("y", f16Zeros 64), ("z", f16Zeros 64), ("w", f16Zeros 64)] []] [(["y", "z", "w"], [AnyNan])]
  , Case "param_to_bf16" (checked "param_to_bf16" paramToBf16) [Inputs [("s", VF32 (castWord32ToFloat s))] [("y", bf16Zeros 32), ("z", zeros 32)] [] | s <- narrowed] [(["y"], [AnyNan])]
  , Case "param_to_bf16_once" (checked "param_to_bf16_once" paramToBf16Once) [Inputs [("s", VF32 (castWord32ToFloat s))] [("y", bf16Zeros 32)] [] | s <- narrowed] [(["y"], [AnyNan])]
  , Case "param_to_bf16_var" (checked "param_to_bf16_var" paramToBf16Var) [Inputs [("s", VF32 (castWord32ToFloat s))] [("x", f32s 104 32), ("y", bf16Zeros 32), ("z", zeros 32)] [] | s <- narrowed] [(["y", "z"], [AnyNan])]
  , Case "both_narrowings" (checked "both_narrowings" bothNarrowings) [Inputs [("n", u 64), ("s", VF32 (castWord32ToFloat s))] [("yb", bf16Zeros 64), ("yh", f16Zeros 64)] [] | s <- narrowed] [(["yb", "yh"], [AnyNan])]
  , Case "narrow_mixed" (checked "narrow_mixed" narrowMixed) [Inputs [("s", VF32 (castWord32ToFloat s))] [("x", f32s 107 128), ("o_g", bf16Zeros 128), ("o_s", bf16Zeros 128), ("o_t", bf16Zeros 128), ("o_p", bf16Zeros 128), ("o_h", f16Zeros 128), ("o_f", zeros 128)] [] | s <- narrowed] [(["o_g", "o_s", "o_t", "o_p", "o_h"], [AnyNan, Ftz])]
  , Case "select_bf16" (checked "select_bf16" selectBf16) [Inputs [] [("a", [VBF16 (BF16 w) | w <- bf16Bits]), ("c", choose), ("y", bf16Zeros 32)] []] []
  , Case "select_f16_f32" (checked "select_f16_f32" selectF16F32) [Inputs [] [("a", [VF16 (F16 w) | w <- take 64 (cycle f16Specials)]), ("b", map (VF32 . castWord32ToFloat) (take 64 (cycle f32Specials))), ("c", choose), ("y", f16Zeros 32), ("z", zeros 32)] []] [(["z"], [Ftz, Operand])]
  , Case "udiv_literal" (checked "udiv_literal" udivLiteral) [Inputs [] [("d", map VU32 divisors), ("q", replicate 64 (VU32 0)), ("r", replicate 64 (VU32 0))] []] []
  , Case "f16_div_literal" (checked "f16_div_literal" f16DivLiteral) [Inputs [] [("x", [VF16 (F16 h) | h <- [0 .. 65535]]), ("y", f16Zeros 65536)] []] [(["y"], [NanClass])]
  , Case "sqrt_widened" (checked "sqrt_widened" sqrtWidened) [Inputs [] ([("h", [VF16 (F16 h) | h <- [0 .. 65535]]), ("b", [VBF16 (BF16 h) | h <- [0 .. 65535]])] <> [(o, zeros 65536) | o <- ["hs", "hr", "bs", "br"]]) []] [(["hs", "hr", "bs", "br"], [NanClass, Ftz])]
  , Case "f16_select" (checked "f16_select" f16Select) [Inputs [] [("x", [VF16 (F16 h) | h <- [0 .. 65535]]), ("y", [VF16 (F16 (h * 40503 + 12345)) | h <- [0 .. 65535]]), ("o", f16Zeros 65536)] []] []
  ]
 where
  f16Zeros n = replicate n (VF16 (F16 0))
  -- Below 65520 but for a few, so that the literal is what narrows: f16's
  -- infinity, which the miscompile gives as 0x7bff.
  big = map castWord32ToFloat [0xF149F2CA, 0x7149F2CA, 0x477FE000, 0x477FF000, 0x477FF001, 0xC77FF000, 0x7F800000, 0xFF800000, 0x7FC00000] <> floats 105 55
  narrowed = [0x3F000000, 0x40490FDB, 0xC0100000, 0x477FF000, 0x7F7FFFFF, 0x7FC00000, 0x7F800000, 0x3F808000]
  bf16Bits = [0x7F81, 0xFF81, 0x7FC1, 0xFFFF, 0x0001, 0x8001, 0x007F, 0x807F, 0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x4049, 0x0040] <> [fromIntegral (w .&. 0xFFFF) | w <- randoms 106 48]
  choose = [VBool (even l || l `mod` 3 == 0) | l <- [0 .. 31 :: Int]]
  divisors = [1 .. 32] <> [33, 47, 64, 97, 127, 128, 255, 256, 1000, 4093, 65535, 65536, 65537, 1000003, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xC0000000, 0xFFFFFF7F, 0xFFFFFF80, 0xFFFFFF81, 0xFFFFFFFE, 0xFFFFFFFF, 3, 5, 7, 9, 11, 13, 17, 19, 23]

------------------------------------------------------------------------------
-- Rules

-- | An output's rule, but its bound: the leaves its case gives it, in the runner's words.
rule :: Case -> Name -> Text
rule c o = T.intercalate "+" ("bits" : map word (nub (sort (leavesOf c o))))
 where
  word = \case
    NanClass -> "class"
    AnyNan -> "nan"
    Ftz -> "ftz"
    Operand -> "operand"
    Zeros -> "zeros"
    Ulp n -> "ulp" <> T.pack (show n)

leavesOf :: Case -> Name -> [Leave]
leavesOf c o = concat [ls | (os, ls) <- c.caseLeaves, o `elem` os]

-- | The cases of products of fragments, compared within a bound.
fragmentCases :: [Text]
fragmentCases = ["gemm_bf16", "gemm_f16", "gemm_f32", "staged", "scaled_rows", "tiled", "tiled_narrow", "gemm_via", "gemm_specials"]

------------------------------------------------------------------------------
-- The corpus

main :: IO ()
main = do
  [root] <- getArgs
  forM_ (cases <> metalCases <> packCases <> coverageCases <> reproCases) $ \c -> do
    let k = checkedIR c.caseKernel
    l <- either (\fs -> fail (T.unpack c.caseId <> ": " <> show fs)) pure (backendLower metal [c.caseKernel])
    t <- case l.loweredKernels of
      [t] -> pure t
      _ -> fail (T.unpack c.caseId <> ": not one launch table")
    forM_ [(i, byte, given) | (i, given) <- zip [0 :: Int ..] c.caseRuns, byte <- poisons] $ \(i, byte, given) -> do
      let inp = poisoned k byte given
          name = T.unpack c.caseId <> "_" <> show i <> "_" <> showHex byte ""
          dir = root <> "/cases/" <> name
          file f = dir <> "/" <> f
          runOf kern ins = case runIRWithTables (targetCaps @AppleGPU9) kern (withTables (Map.fromList ins.inTables) (withSeed 7 (launch (Map.fromList ins.inParams) (Map.fromList ins.inBuffers)))) of
            Left e -> error (name <> ": the interpreter: " <> show e)
            Right r -> r
          (out, tabs) = runOf k inp
          -- binary32 and bfloat16 subnormals anywhere a target may flush them.
          flushedKernel = flushLiterals k
          subnormal = any (isSubnormal . snd) inp.inParams || any (any isSubnormal . snd) inp.inBuffers || any (any (any isSubnormal) . snd) inp.inTables || flushedKernel /= k
          alt = if subnormal then Just (runOf flushedKernel (mapInputs flushValue inp)) else Nothing
          magnitudes = if c.caseId `elem` fragmentCases then Just (runOf k (mapInputs absValue inp)) else Nothing
          kk = case lookup "k" inp.inParams of
            Just (VU32 n) -> fromIntegral n
            _ -> 64 :: Double
          index n = fromMaybe (error (name <> ": no binding " <> T.unpack n)) (lookup n t.lkBindings)
          bin f vs = BS.writeFile (file f) (bytes vs)
          -- An array's columns: type, bytes, input, and its output's file, the
          -- interpreter's run on flushed inputs where its rule takes that,
          -- and its rule, or dashes.
          columns what ty vs access label output altOut mags = do
            let inFile = "in_" <> label <> ".bin"
            bin inFile vs
            outCols <- case (access, output) of
              (IR.ReadOnly, _) -> pure ["-", "-", "-"]
              (_, Nothing) -> pure ["-", "-", "-"]
              (_, Just o) -> do
                bin ("out_" <> label <> ".bin") o
                altCol <- case altOut of
                  Just a | Ftz `elem` leavesOf c what -> ("out2_" <> label <> ".bin") <$ bin ("out2_" <> label <> ".bin") a
                  _ -> pure "-"
                r <- case mags of
                  Just ms -> do
                    BS.writeFile (file ("bound_" <> label <> ".bin")) (BS.concat [f64le (boundOf kk m) | m <- ms])
                    pure (T.unpack (rule c what) <> "+bound:bound_" <> label <> ".bin")
                  Nothing -> pure (T.unpack (rule c what))
                pure ["out_" <> label <> ".bin", altCol, r]
            pure (unwords ([typeName ty, show (length vs * fromIntegral (scalarSize ty)), inFile] <> outCols))
      createDirectoryIfMissing True dir
      T.writeFile (file "source.metal") l.loweredSource
      bufferLines <- forM k.kernelBuffers $ \b -> do
        let vs = fromMaybe (error (name <> ": no buffer " <> T.unpack b.bufName)) (lookup b.bufName inp.inBuffers)
            label = T.unpack b.bufName
        rest <- columns b.bufName b.bufType vs b.bufAccess label (Map.lookup b.bufName out) (alt >>= Map.lookup b.bufName . fst) (magnitudes >>= Map.lookup b.bufName . fst)
        pure ("buffer " <> show (index b.bufName) <> " " <> label <> " " <> rest)
      tableLines <- forM k.kernelTables $ \tb -> do
        let es = fromMaybe [] (lookup tb.tableName inp.inTables)
            access = case tb.tableAccess of
              IR.ReadOnly -> "read"
              IR.WriteOnly -> "write"
              _ -> "read_write"
            entry m j = m >>= Map.lookup tb.tableName . snd >>= \xs -> if j < length xs then Just (xs !! j) else Nothing
        arrays <- forM [(j, vs) | (j, vs) <- zip [0 :: Int ..] es, not (null vs)] $ \(j, vs) -> do
          let label = "tbl_" <> T.unpack tb.tableName <> "_" <> show j
          rest <- columns tb.tableName tb.tableType vs tb.tableAccess label (entry (Just (out, tabs)) j) (entry alt j) (entry magnitudes j)
          pure ("array " <> T.unpack tb.tableName <> " " <> show j <> " " <> rest)
        pure (("table " <> show (index tb.tableName) <> " " <> T.unpack tb.tableName <> " " <> access <> " " <> show (length es)) : arrays)
      paramsLine <- case t.lkParamsBinding of
        Nothing -> pure []
        Just p -> do
          BS.writeFile (file "params.bin") (paramBytes (abiOf k) inp.inParams)
          pure ["params " <> show p <> " params.bin"]
      writeFile (file "case.txt") . unlines $
        [ "entry " <> T.unpack t.lkEntry
        , "threads " <> show t.lkThreads
        , "groups " <> show (groupsOf inp.inParams t.lkGroups)
        , "tg_memory " <> show t.lkTgMemory
        , "simd " <> show k.kernelSimdWidth
        ]
          <> paramsLine
          <> bufferLines
          <> concat tableLines

{- | A fragment product's bound from the sum of its products' magnitudes:
(k + 2) · 2^-24 of it, and the least normal, 2^-126.
-}
boundOf :: Double -> Value -> Double
boundOf kk = \case
  VF32 m -> (kk + 2) * 2 ** (-24) * realToFrac (abs m) + 2 ** (-126)
  _ -> 0

f64le :: Double -> BS.ByteString
f64le d = BS.pack [fromIntegral (w `shiftR` s) | s <- [0, 8 .. 56]] where w = castDoubleToWord64 d

-- | The grid's threadgroups for a run's parameters: dimensions, u32 literals and + - * / %.
groupsOf :: [(Name, Value)] -> Expr -> Word32
groupsOf ps = go
 where
  go = \case
    ELit (LU32 n) -> n
    EParam n | Just (VU32 v) <- lookup n ps -> v
    EBin op a b -> case op of
      IR.Add -> go a + go b
      IR.Sub -> go a - go b
      IR.Mul -> go a * go b
      IR.Div -> go a `div` go b
      IR.Rem -> go a `mod` go b
      _ -> error "a grid of another operation"
    e -> error ("a grid of " <> show e)

typeName :: ScalarType -> String
typeName = \case
  TF32 -> "f32"
  TF16 -> "f16"
  TBF16 -> "bf16"
  TI32 -> "i32"
  TU32 -> "u32"
  TI8 -> "i8"
  TU8 -> "u8"
  TBool -> "bool"

-- | A run's parameters, buffers and tables, each value mapped.
mapInputs :: (Value -> Value) -> Inputs -> Inputs
mapInputs f i =
  i
    { inParams = [(n, f v) | (n, v) <- i.inParams]
    , inBuffers = [(n, map f vs) | (n, vs) <- i.inBuffers]
    , inTables = [(n, map (map f) es) | (n, es) <- i.inTables]
    }

{- | The poisons a run's write-only arrays start as, every byte the same:
0xff, a NaN in each float type and -1 in each integer one, and 0x5a, a
number in each; for a bool, true and false. Neither is a subnormal, which
the flushed run would change.
-}
poisons :: [Word8]
poisons = [0xff, 0x5a]

{- | A run's write-only buffers, and the arrays of its write-only tables,
as a poison, in the interpreter's run as on the GPU.
-}
poisoned :: IR.Kernel -> Word8 -> Inputs -> Inputs
poisoned k p i =
  i
    { inBuffers = [(n, maybe vs (\ty -> map (const (poison p ty)) vs) (lookup n buffers)) | (n, vs) <- i.inBuffers]
    , inTables = [(n, maybe es (\ty -> map (map (const (poison p ty))) es) (lookup n tables)) | (n, es) <- i.inTables]
    }
 where
  buffers = [(b.bufName, b.bufType) | b <- k.kernelBuffers, b.bufAccess == IR.WriteOnly]
  tables = [(t.tableName, t.tableType) | t <- k.kernelTables, t.tableAccess == IR.WriteOnly]

-- | A value of a type whose every byte is @p@; a bool true where @p@ is odd.
poison :: Word8 -> ScalarType -> Value
poison p = \case
  TF32 -> VF32 (castWord32ToFloat w32)
  TF16 -> VF16 (F16 w16)
  TBF16 -> VBF16 (BF16 w16)
  TI32 -> VI32 (fromIntegral w32)
  TU32 -> VU32 w32
  TI8 -> VI8 (fromIntegral p)
  TU8 -> VU8 p
  TBool -> VBool (odd p)
 where
  w16 = fromIntegral p * 0x0101 :: Word16
  w32 = fromIntegral p * 0x01010101 :: Word32

isSubnormal :: Value -> Bool
isSubnormal = \case
  VF32 x -> let w = castFloatToWord32 x in w .&. 0x7F800000 == 0 && w .&. 0x007FFFFF /= 0
  VBF16 (BF16 w) -> w .&. 0x7F80 == 0 && w .&. 0x007F /= 0
  _ -> False

-- | A binary32 or bfloat16 subnormal as a zero of its sign.
flushValue :: Value -> Value
flushValue v = case v of
  VF32 x | isSubnormal v -> VF32 (if castFloatToWord32 x >= 0x80000000 then -0 else 0)
  VBF16 (BF16 w) | isSubnormal v -> VBF16 (BF16 (w .&. 0x8000))
  _ -> v

{- | A kernel's binary32 and bfloat16 subnormal literals as zeros of their
signs, for the run on flushed inputs: a target may flush a literal
operand as it does a loaded one.
-}
flushLiterals :: IR.Kernel -> IR.Kernel
flushLiterals k = k {IR.kernelBody = map stmt k.kernelBody}
 where
  stmt = \case
    IR.SLet n ty e -> IR.SLet n ty (expr e)
    IR.SVar n ty e -> IR.SVar n ty (expr e)
    IR.SAssign n e -> IR.SAssign n (expr e)
    IR.SStore b i e -> IR.SStore b (expr i) (expr e)
    IR.SStoreVia tb en i e -> IR.SStoreVia tb (expr en) (expr i) (expr e)
    IR.STgStore a i e -> IR.STgStore a (expr i) (expr e)
    IR.SLoadPack ns ty b i -> IR.SLoadPack ns ty b (expr i)
    IR.SStorePack b i es -> IR.SStorePack b (expr i) (map expr es)
    IR.STgStorePack a i es -> IR.STgStorePack a (expr i) (map expr es)
    IR.SLoadPackVia ns ty tb en i -> IR.SLoadPackVia ns ty tb (expr en) (expr i)
    IR.SIf r cond th el -> IR.SIf r (expr cond) (map stmt th) (map stmt el)
    IR.SFor r n from to step body -> IR.SFor r n (expr from) (expr to) (expr step) (map stmt body)
    IR.SBarrier -> IR.SBarrier
    IR.SSimdBarrier -> IR.SSimdBarrier
    IR.SSimd n op ty e -> IR.SSimd n op ty (expr e)
    IR.SFragVar n ty -> IR.SFragVar n ty
    IR.SFragLoad f m i stride tr -> IR.SFragLoad f (memory m) (expr i) (expr stride) tr
    IR.SFragMma d a b acc -> IR.SFragMma d a b acc
    IR.SFragConvert d s -> IR.SFragConvert d s
    IR.SFragScaleRows f a i -> IR.SFragScaleRows f a (expr i)
    IR.SFragStore m i stride f -> IR.SFragStore (memory m) (expr i) (expr stride) f
  memory = \case
    IR.FVia tb en -> IR.FVia tb (expr en)
    m -> m
  expr = \case
    ELit l -> ELit (literal l)
    EUn op a -> EUn op (expr a)
    EBin op a b -> EBin op (expr a) (expr b)
    ESelect cond a b -> ESelect (expr cond) (expr a) (expr b)
    EFma a b acc -> EFma (expr a) (expr b) (expr acc)
    EConvert ty a -> EConvert ty (expr a)
    ENarrowInt o ty a -> ENarrowInt o ty (expr a)
    ELoad b i -> ELoad b (expr i)
    ELoadVia tb en i -> ELoadVia tb (expr en) (expr i)
    ETgLoad a i -> ETgLoad a (expr i)
    e@(EVar _) -> e
    e@(EParam _) -> e
    e@(EBuiltin _) -> e
  literal = \case
    LF32 x | isSubnormal (VF32 x) -> LF32 (if castFloatToWord32 x >= 0x80000000 then -0 else 0)
    LBF16 w | isSubnormal (VBF16 (BF16 w)) -> LBF16 (w .&. 0x8000)
    l -> l

absValue :: Value -> Value
absValue = \case
  VF32 x -> VF32 (abs x)
  VBF16 (BF16 w) -> VBF16 (BF16 (w .&. 0x7FFF))
  VF16 (F16 w) -> VF16 (F16 (w .&. 0x7FFF))
  v -> v

paramBytes :: Abi -> [(Name, Value)] -> BS.ByteString
paramBytes abi ps =
  BS.pack [Map.findWithDefault 0 i placed | i <- [0 .. fromIntegral abi.abiParamsSize - 1 :: Int]]
 where
  placed = Map.fromList (concat [zip [fromIntegral f.fieldOffset ..] (maybe [] valueBytes (lookup f.fieldName ps)) | f <- abi.abiParams])

bytes :: [Value] -> BS.ByteString
bytes = BS.pack . concatMap valueBytes

valueBytes :: Value -> [Word8]
valueBytes = \case
  VF32 x -> le32 (castFloatToWord32 x)
  VF16 (F16 w) -> le16 w
  VBF16 (BF16 w) -> le16 w
  VI32 n -> le32 (fromIntegral n :: Word32)
  VU32 n -> le32 n
  VI8 n -> [fromIntegral n]
  VU8 n -> [n]
  VBool b -> [if b then 1 else 0]
 where
  le16 w = [fromIntegral w, fromIntegral (w `shiftR` 8)]
  le32 w = [fromIntegral (w `shiftR` s) | s <- [0, 8, 16, 24 :: Int]]
