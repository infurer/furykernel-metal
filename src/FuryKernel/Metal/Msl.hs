-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Checked kernels as Metal Shading Language: one deterministic
translation unit for a set of kernels of distinct names, and their launch
tables.

The source keeps the author's names. Every library function is called
fully qualified, so that no name of the author's can shadow one, and the
generated names start with @fk_@, which the gate reserves. Qualification
can't escape a macro: Metal's predefined macros, and its typedefs as a
kernel's name, are the core's gate's to refuse ("FuryKernel.Check").

What the printing fixes:

* finite f32 literals are written as their shortest decimals that read
  back to the same value; f32 infinities and NaNs, and every f16 and bf16
  literal, through @as_type@ from their bits;
* select is @metal::select@, which evaluates both sides as the IR says,
  a half's on its bits: under Metal's default options, fast math among
  them, a half select of a comparison, a minimum's idiom, ignored a NaN
  operand as @fmin@ does, despite the source's math mode (observed on
  macOS 27.2's compiler);
  And and Or are @bool(int(a) & int(b))@ and @|@, both operands
  evaluated;
* SIMD-group sums are @metal::simd_sum@, which adds a group's lanes in
  the IR's order, pairwise, neighbours first, as observed on an M3 Max
  (AppleGPU9), MSL leaving the order unspecified;
* a SIMD-group barrier is @metal::simdgroup_barrier@ over threadgroup
  memory;
* an integer narrows clamped with @metal::clamp@ or @metal::min@, or
  wrapped through @uint@ and @as_type@;
* i32 addition, subtraction, multiplication, negation and magnitude go
  through @uint@, so that they wrap as the IR defines them, the magnitude
  in @fk_abs_i32@, which takes its operand once;
* a u32 quotient or remainder by a divisor that isn't a literal is
  @fk_udiv@'s or @fk_urem@'s, which halve the dividend first and correct
  once: macOS 27.2's compiler got quotients of dividends near 2^32 wrong
  (@4294967168u % 1@ came out 4294967170);
* half's negation and magnitude go through its bits, as IEEE 754 defines
  them: Metal's own give a NaN the canonical one (observed on an M3 Max,
  AppleGPU9);
* float's magnitude goes through its bits too: macOS 27.2's compiler drops
  @metal::abs@ where it takes the operand for non-negative, so that
  @abs(rsqrt(-0))@ came out @-inf@; float's negation it keeps;
* float minima and maxima, and the SIMD-group reductions over them, are
  Metal's own, whose choices among equal operands and NaNs the IR leaves
  to a target: binary32's the precise @metal::precise::fmin@ and
  @metal::precise::fmax@, which hold whatever the compile options;
  binary16's @metal::fmin@ and @metal::fmax@, as MSL has no precise half
  overloads, held by the source's math mode and the required options;
  and @metal::simd_min@ and @metal::simd_max@;
* every threadgroup array is declared @alignas(16)@, so that Metal starts
  each at a multiple of 16 bytes; the kernel's layout
  ('FuryKernel.IR.tgLayout'), each one's size rounded up to 16, summed,
  the launch table's 'FuryKernel.Backend.lkTgMemory', is the most Metal
  allocates: as observed on an M3 Max (AppleGPU9), it allocates that where
  the kernel uses every array, and less where it doesn't, dropping the
  arrays a kernel doesn't use and shrinking some it indexes by constants;
* transcendental functions are @metal::precise::@ ones, so results don't
  depend on the compile options' math mode, and within MSL's bounds for
  them (the Metal Shading Language Specification 4.1, Table 8.1): exp and
  log 4 ulp, tanh 5, sqrt and rsqrt correctly rounded. A float's sqrt and
  rsqrt take their operand through @fk_keep@ (@copysign(x + 0.0f, x)@):
  macOS 27.2's compiler didn't round them correctly for a value widened
  from half or bfloat in the kernel, and past @fk_keep@ they were
  observed correctly rounded for every half and every bfloat;
* binary16 math functions and division compute in binary32 and round
  once: exp and log within half a binary16 ulp plus 4 binary32 ulp of the
  exact value, tanh plus 5, rsqrt plus half a binary32 ulp, sqrt and
  division correctly rounded, binary32 having twice binary16's bits and
  two more. Division doubles both operands first, which is exact: macOS
  27.2's compiler rounded a quotient by a constant divisor wrongly (by
  every one but a power of two: 73.3M of 269M pairs of 4104 constant
  divisors and every half), and with the operands doubled every quotient
  was observed correctly rounded (every half by those 4104 divisors, and
  every pair of halves at run time); binary16 addition, subtraction and
  multiplication stay native, being IEEE operations on the target;
* the source sets its own math mode, @#pragma METAL fp math_mode(safe)@,
  and refuses MSL before 4.1. With fast math on, the pragma kept a NaN's
  comparison with itself true, but not half's select of a comparison
  (above) from ignoring a NaN (observed on an M3 Max, AppleGPU9, under
  macOS 27.2), so the lowered options stay required ("FuryKernel.Metal");
* a fused multiply-add is @metal::fma@, which MSL defines as correctly
  rounded in float and half (Tables 8.1 and 8.3); the source's
  @#pragma METAL fp contract(off)@, after the math mode, which sets the
  contract on, keeps the compiler from fusing any
  other multiply and add, which MSL's safe math mode would within a
  statement, so a multiply-add is fused only where a kernel writes fma;
* binary32 subnormals are left to the target by the IR, and MSL lets them
  be flushed to zero (section 8.1). On the M3 Max, float arithmetic
  (addition, subtraction, multiplication, division, sqrt, rsqrt, fma and
  simd_sum) takes a subnormal operand as a zero of its sign and gives one
  for a subnormal result (fma for one tiny before rounding too, even where
  it rounds to the least normal), as do exp for its results and log for
  its operands; comparisons see subnormals as zeros, and so do fmin,
  fmax, select's condition and simd_max, which return an operand's bits
  unflushed, simd_max over subnormal lanes giving lane 0's; loads,
  stores, copies, bfloat's widening and tanh of a subnormal keep the
  bits, and the narrowing to bfloat, by bits, rounds one as IEEE 754
  does. half keeps its subnormals, fma's included, as the IR requires,
  binary16 having no such leave; MSL lets them be flushed too, so the
  target meets the IR there by this observation alone;
* a NaN converts as the target has it, which the IR allows: as observed on
  an M3 Max (AppleGPU9), bfloat's widening keeps a signalling NaN
  signalling, half's widening and float's narrowing to half give the
  canonical NaN, and no NaN becomes an infinity; the narrowing to bfloat,
  by bits, quiets a NaN, keeping its sign and the top of its payload;
* conversions are @static_cast@, which rounds to nearest with ties to
  even under the compile options' default conversion rounding
  ("FuryKernel.Metal"): MSL fixes that for float's narrowing to half
  (section 8.6), and an integer's conversion to float, which MSL lets
  round toward zero too, rounded to nearest even as observed on an M3 Max
  (AppleGPU9). Float's narrowing to half takes its operand through
  @fk_keep@: macOS 27.2's compiler narrowed a maximum, minimum or select
  of a constant toward zero, saturating;
* a bf16 value that a select, a var or a literal gives is held as its
  bits, a @ushort@, and widened by them (@as_type<float>(uint(u) << 16)@):
  Metal's compiler miscompiled selects of bfloat; every other bf16 value
  is a @bfloat@, loaded, stored, packed and widened natively;
  every narrowing to bf16 goes through @fk_bf16@, which rounds to
  nearest with ties to even by bits: Metal's narrowing of a value the same
  in every thread of a launch, narrowed to f16 too, gave the f16 bits
  (observed on macOS 27.2's compiler);
* a parameter struct's size and each field's offset are asserted against
  the ABI's (@static_assert@), the layout runtimes fill it by;
* a pack is one load or store of a @ushort@, @ushort2@ (two 16-bit
  elements), @uint@, @uint2@ or @uint4@ through the buffer's pointer, its
  elements taken apart and put together with @as_type@, so every element
  type and width lowers the same way; a pack of four-byte elements stored
  to a threadgroup array is stored as their own vector (@float2@,
  @float4@, @int4@ and the like), narrower ones as words;
* those words are named through typedefs that may alias every type
  (@fk_uint4@ for @uint4@ and the like, declared with the @may_alias@
  attribute), at every language version, where the kernel writes the
  memory (a writable buffer, a threadgroup array, a table written
  through), since a pack's memory is accessed as its elements' own type
  too: as observed on macOS 27.2's compiler at MSL 4.1, type-based alias
  analysis otherwise takes a @uint4@ access and a @float@ one, say, to be
  of different objects, so that a pack reloaded after an element's store
  reads the old value, as does an element reloaded after a pack's store
  (at 3.1 it takes vectors, though not scalars, to alias every type; at
  4.1 a vector of the elements' own type, as threadgroup memory's
  four-byte packs are stored, aliases them). A pack of a buffer or table
  the kernel only reads keeps the plain type: the IR's bindings never
  overlap, so no store of the kernel's can touch that memory;
* a table is a buffer of @ulong@ GPU addresses, each cast to a pointer to
  its arrays' element type where it is read;
* a fragment is a @metal::simdgroup_matrix@ of 8×8, made zero with
  @make_filled_simdgroup_matrix@, loaded and stored with @simdgroup_load@
  and @simdgroup_store@ from a pointer to its first element, the stride as
  the elements per row (a transposed load reads the block transposed), and
  multiplied with @simdgroup_multiply_accumulate@, bfloat or half into
  float, a product MSL documents for one element type only, both observed
  on an M3 Max (AppleGPU9);
* a fragment widens into float a lane's elements at a time
  (@thread_elements()@), taking a bfloat or half fragment's element k to
  sit where a float fragment's does, which MSL doesn't specify: observed
  on an M3 Max (AppleGPU9) from both;
* a fragment's rows are scaled a lane's elements at a time too, each
  lane's row found from the 8×8 layout observed on an M3 Max (AppleGPU9),
  which MSL doesn't specify: a lane holds two elements of row
  @(lane / 4 & 4) + (lane / 2 % 4)@.

'metal' refuses what it doesn't print yet, a finding of the Target rule
for each: tables written through and fragments loaded or stored through
tables.

Buffers bind in declaration order from index 0, tables after them, the
parameter struct last. The source is MSL 4.1, the floor: a language
version is the operating system's, not the GPU's, and 4.1 is macOS 27's.
-}
module FuryKernel.Metal.Msl (
  metal,
  mslLanguage,
) where

import Control.Applicative ((<|>))
import Control.Monad.Trans.State.Strict (State, get, gets, modify', runState)
import Data.Containers.ListUtils (nubOrd)
import Data.Foldable (traverse_)
import Data.List (sortOn)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Abi (abiOf, hex64, irHash)
import FuryKernel.Abi qualified as Abi
import FuryKernel.Backend (Backend (..), CompileOptions (..), Lowered (..), LoweredKernel (..), compileOptions)
import FuryKernel.Check (Checked, checkedIR)
import FuryKernel.Finding (Finding (..), Location (..), Rule (Names, Target), locatedAt)
import FuryKernel.IR hiding (Kernel (..))
import FuryKernel.IR qualified as IR
import FuryKernel.Metal.Target (AppleGPU9)
import FuryKernel.Type (ScalarType (..), scalarSize)
import GHC.Float (castFloatToWord32)
import Numeric (showHex)

-- | The language the source is in; its version is in the compile options.
mslLanguage :: Text
mslLanguage = "msl"

{- | The Metal target's backend: checked kernels for 'AppleGPU9' as one
MSL translation unit, in name order, the options to compile it with, and
each kernel's launch table. It refuses two kernels of one name, which
would declare one entry point twice.
-}
metal :: Backend AppleGPU9
metal =
  Backend
    { backendName = "metal"
    , backendLower = \ks -> case repeats (map checkedIR ks) <> concatMap refusals ks of
        [] -> Right (lower ks)
        findings -> Left findings
    }

-- | Kernels of one name, which would declare one entry point twice.
repeats :: [IR.Kernel] -> [Finding]
repeats ks =
  [ Finding Names ("The kernel " <> n <> " is lowered " <> showT c <> " times into one translation unit.")
  | (n, c) <- Map.toList (Map.fromListWith (+) [(k.kernelName, 1 :: Int) | k <- ks])
  , c > 1
  ]

{- | A finding for each construct of a kernel this lowering doesn't print
yet ('notLowered').
-}
refusals :: Checked t -> [Finding]
refusals c =
  [ locatedAt (Location k.kernelName []) (Finding Target ("FuryKernel.Metal doesn't lower " <> what <> " yet."))
  | what <- nubOrd (notLowered k)
  ]
 where
  k = checkedIR c

{- | The constructs of a kernel this lowering doesn't print yet: tables
written through and fragments loaded or stored through tables.
-}
notLowered :: IR.Kernel -> [Text]
notLowered k = ["tables written through" | any ((/= ReadOnly) . (.tableAccess)) k.kernelTables] <> concatMap construct k.kernelBody
 where
  construct = \case
    SFragLoad _ FVia {} _ _ _ -> ["fragments loaded or stored through tables"]
    SFragStore FVia {} _ _ _ -> ["fragments loaded or stored through tables"]
    SIf _ _ th el -> concatMap construct (th <> el)
    SFor _ _ _ _ _ body -> concatMap construct body
    _ -> []

{- | What the printer makes of a construct 'refusals' keeps from it: never
reached.
-}
notPrinted :: a
notPrinted = error "FuryKernel.Metal.Msl: a construct refused before printing"

lower :: [Checked t] -> Lowered
lower ks =
  Lowered
    { loweredLanguage = mslLanguage
    , loweredOptions = compileOptions {optionsLanguageVersion = Just "4.1"}
    , loweredSource = renderMsl ks
    , loweredKernels = map table ks
    }
 where
  table c =
    LoweredKernel
      { lkName = k.kernelName
      , lkEntry = k.kernelName
      , lkAbi = abiOf k
      , lkIrHash = irHash c
      , lkThreads = k.kernelThreads
      , lkGroups = k.kernelGroups
      , lkBindings = zip (map (.bufName) k.kernelBuffers <> map (.tableName) k.kernelTables) [0 ..]
      , lkParamsBinding = if null k.kernelParams then Nothing else Just (fromIntegral (length k.kernelBuffers + length k.kernelTables))
      , lkTgMemory = snd (tgLayout k.kernelTgArrays)
      }
   where
    k = checkedIR c

-- | The translation unit for a set of kernels, in name order.
renderMsl :: [Checked t] -> Text
renderMsl ks =
  T.unlines $
    [ "// Generated by FuryKernel's Metal target. Do not edit: regenerate."
    , "#include <metal_stdlib>"
    , "#if __METAL_VERSION__ < 410"
    , "#error \"FuryKernel's Metal target needs MSL 4.1 (macOS 27).\""
    , "#endif"
    , -- The IR's arithmetic whatever the compile options: fast math folds a
      -- NaN's comparison with itself to false.
      "#pragma METAL fp math_mode(safe)"
    , -- Safe math mode sets the FP contract on, a * b + c in one statement
      -- fusing, so this comes after it. Off, a multiply-add is fused only
      -- where a kernel writes fma.
      "#pragma METAL fp contract(off)"
    , "using namespace metal;"
    ]
      <> concatMap helperSource (Set.toList (Set.unions (map snd kernels)))
      <> concatMap fst kernels
 where
  kernels = map kernelSource (sortOn ((.kernelName) . checkedIR) ks)

{- | What the printing declares ahead of the kernels: a function it calls
rather than repeat an operand, or a pack's container under a name that
may alias.
-}
data Helper
  = {- | @fk_uint4@ and the like: a container that may alias every type, which
    a pack's loads and stores go through.
    -}
    Container Text
  | -- | @fk_abs_i32@: an i32's magnitude, wrapping as the IR defines it.
    AbsI32
  | -- | @fk_bf16@: a float narrowed to bf16's bits.
    NarrowBF16
  | -- | @fk_udiv@ and @fk_urem@: u32 division by a run-time divisor, its dividend halved first.
    UDivU32
  | -- | @fk_keep@: a float as it is, kept from the compiler's look through a widening or a choice.
    Keep
  deriving stock (Eq, Ord, Show)

helperSource :: Helper -> [Text]
helperSource = \case
  Container c ->
    [ ""
    , "typedef " <> c <> " __attribute__((may_alias)) fk_" <> c <> ";"
    ]
  Keep ->
    [ ""
    , "static inline float fk_keep(float x)"
    , "{"
    , "    return metal::copysign(x + 0.0f, x);"
    , "}"
    ]
  UDivU32 ->
    [ ""
    , "static inline uint fk_udiv(uint c, uint d)"
    , "{"
    , "    const uint q = ((c >> 1u) / d) << 1u;"
    , "    return c - q * d >= d ? q + 1u : q;"
    , "}"
    , ""
    , "static inline uint fk_urem(uint c, uint d)"
    , "{"
    , "    const uint r = c - (((c >> 1u) / d) << 1u) * d;"
    , "    return r >= d ? r - d : r;"
    , "}"
    ]
  AbsI32 ->
    [ ""
    , "static inline int fk_abs_i32(int x)"
    , "{"
    , "    return metal::select(x, as_type<int>(0u - as_type<uint>(x)), x < 0);"
    , "}"
    ]
  -- To nearest, ties to even; a NaN quieted, its sign and top payload kept.
  NarrowBF16 ->
    [ ""
    , "static inline ushort fk_bf16(float x)"
    , "{"
    , "    const uint u = as_type<uint>(x);"
    , "    return ushort((u & 0x7fffffffu) > 0x7f800000u ? (u >> 16) | 0x40u : (u + 0x7fffu + ((u >> 16) & 1u)) >> 16);"
    , "}"
    ]

-- | A kernel's source, and the helpers it uses.
kernelSource :: Checked t -> ([Text], Set Helper)
kernelSource c = (source, end.envHelpers)
 where
  (body', end) = runState (concat <$> traverse (stmt 1) k.kernelBody) (initialEnv k)
  source =
    [ ""
    , "// " <> k.kernelName <> ": IR hash " <> hex64 (irHash c) <> ", ABI hash " <> hex64 (Abi.abiHash abi)
    ]
      <> paramsStruct
      <> [ "[[max_total_threads_per_threadgroup(" <> showT k.kernelThreads <> ")]]"
         , "kernel void " <> k.kernelName <> "("
         ]
      <> commaSeparated (map ("    " <>) arguments)
      <> ["{"]
      <> ["    alignas(16) threadgroup " <> mslType a.tgType <> " " <> a.tgName <> "[" <> showT a.tgCount <> "];" | a <- k.kernelTgArrays]
      <> body'
      <> ["}"]
  k = checkedIR c
  abi = abiOf k
  paramsName = "fk_params_" <> k.kernelName
  paramsStruct
    | null k.kernelParams = []
    | otherwise =
        ["struct " <> paramsName <> " {"]
          <> ["    " <> mslType p.paramType <> " " <> p.paramName <> ";" | p <- k.kernelParams]
          <> ["};"]
          -- The layout runtimes fill it by: the ABI's.
          <> ["static_assert(sizeof(" <> paramsName <> ") == " <> showT abi.abiParamsSize <> ", \"" <> paramsName <> ": the ABI's size\");"]
          <> [ "static_assert(__builtin_offsetof(" <> paramsName <> ", " <> f.fieldName <> ") == " <> showT f.fieldOffset <> ", \"" <> paramsName <> "." <> f.fieldName <> ": the ABI's offset\");"
             | f <- abi.abiParams
             ]
          <> [""]
  arguments =
    [ "device " <> constness b <> mslType b.bufType <> "* " <> b.bufName <> " [[buffer(" <> showT i <> ")]]"
    | (b, i) <- zip k.kernelBuffers [0 :: Int ..]
    ]
      <> [ "device const ulong* " <> t.tableName <> " [[buffer(" <> showT i <> ")]]"
         | (t, i) <- zip k.kernelTables [length k.kernelBuffers ..]
         ]
      <> [ "constant " <> paramsName <> "& fk_p [[buffer(" <> showT (length k.kernelBuffers + length k.kernelTables) <> ")]]"
         | not (null k.kernelParams)
         ]
      <> [ "uint3 fk_thread [[thread_position_in_threadgroup]]"
         , "uint3 fk_group [[threadgroup_position_in_grid]]"
         , "uint fk_lane [[thread_index_in_simdgroup]]"
         , "uint fk_simd_group [[simdgroup_index_in_threadgroup]]"
         ]
  constness b = if b.bufAccess == ReadOnly then "const " else ""
  commaSeparated xs = case reverse xs of
    [] -> [")"]
    lastArg : rest -> map (<> ",") (reverse rest) <> [lastArg <> ")"]

------------------------------------------------------------------------------
-- Types, so that the printing can follow them

data Env = Env
  { envScopes :: [Map Name ScalarType]
  , envHelpers :: Set Helper
  {- ^ The helpers the kernel's expressions call and its packs go through,
  so far.
  -}
  , envParams :: Map Name ScalarType
  , envBuffers :: Map Name ScalarType
  , envArrays :: Map Name ScalarType
  , envAccess :: Map Name Access
  , envTables :: Map Name ScalarType
  , envHeld :: Set Name
  -- ^ The bf16 locals held as their bits.
  }

initialEnv :: IR.Kernel -> Env
initialEnv k =
  Env
    { envScopes = [Map.empty]
    , envHelpers = Set.empty
    , envParams = Map.fromList [(p.paramName, p.paramType) | p <- k.kernelParams]
    , envBuffers = Map.fromList [(b.bufName, b.bufType) | b <- k.kernelBuffers]
    , envArrays = Map.fromList [(a.tgName, a.tgType) | a <- k.kernelTgArrays]
    , envAccess = Map.fromList [(b.bufName, b.bufAccess) | b <- k.kernelBuffers]
    , envTables = Map.fromList [(t.tableName, t.tableType) | t <- k.kernelTables]
    , envHeld = Set.empty
    }

declare :: Name -> ScalarType -> State Env ()
declare n t = modify' $ \e -> case e.envScopes of
  top : rest -> e {envScopes = Map.insert n t top : rest}
  [] -> e {envScopes = [Map.singleton n t]}

scoped :: State Env a -> State Env a
scoped m = do
  modify' (\e -> e {envScopes = Map.empty : e.envScopes})
  r <- m
  modify' (\e -> e {envScopes = drop 1 e.envScopes})
  pure r

-- | The type of a checked kernel's expression; the gate made it total.
typeOf :: Env -> Expr -> ScalarType
typeOf env = \case
  ELit l -> literalType l
  EVar n -> fromMaybe TF32 (foldr (\m acc -> Map.lookup n m <|> acc) Nothing env.envScopes)
  EParam n -> Map.findWithDefault TU32 n env.envParams
  EBuiltin _ -> TU32
  EUn _ e -> typeOf env e
  EBin op a _
    | op `elem` [Eq, Ne, Lt, Le, Gt, Ge, And, Or] -> TBool
    | otherwise -> typeOf env a
  ESelect _ a _ -> typeOf env a
  EFma a _ _ -> typeOf env a
  EConvert t _ -> t
  ENarrowInt _ t _ -> t
  ELoad n _ -> Map.findWithDefault TF32 n env.envBuffers
  ELoadVia n _ _ -> Map.findWithDefault TF32 n env.envTables
  ETgLoad n _ -> Map.findWithDefault TF32 n env.envArrays

------------------------------------------------------------------------------
-- Statements

stmt :: Int -> Stmt -> State Env [Text]
stmt depth = \case
  SLet n t e -> do
    env <- get
    -- A select's, var's or literal's bf16 value stays in its bits.
    let asBits = held env e
    s <- line ("const " <> (if asBits then "ushort" else mslType t) <> " " <> n <> " = ") e ";"
    declare n t
    modify' $ \st -> st {envHeld = if asBits then Set.insert n st.envHeld else st.envHeld}
    pure s
  -- A bf16 var holds its bits.
  SVar n TBF16 e -> do
    e' <- exBits e
    declare n TBF16
    modify' $ \st -> st {envHeld = Set.insert n st.envHeld}
    pure [pad <> "ushort " <> n <> " = " <> e' <> ";"]
  SVar n t e -> do
    s <- line (mslType t <> " " <> n <> " = ") e ";"
    declare n t
    pure s
  SAssign n e -> do
    asBits <- gets (Set.member n . (.envHeld))
    e' <- if asBits then exBits e else ex e
    pure [pad <> n <> " = " <> e' <> ";"]
  SStore b i e -> do
    i' <- ex i
    e' <- exNative e
    pure [pad <> b <> "[" <> i' <> "] = " <> e' <> ";"]
  STgStore a i e -> do
    i' <- ex i
    e' <- exNative e
    pure [pad <> a <> "[" <> i' <> "] = " <> e' <> ";"]
  SLoadPack ns t b i -> do
    i' <- ex i
    access <- gets (Map.findWithDefault ReadOnly b . (.envAccess))
    traverse_ (`declare` t) ns
    let p = packLayout t (length ns)
        raw = "fk_pack_" <> T.concat (take 1 ns)
    c <- if access == ReadOnly then pure p.packContainer else aliased p.packContainer
    let pointer = "reinterpret_cast<device " <> (if access == ReadOnly then "const " else "") <> c <> "*>(" <> b <> ")"
    pure $
      [pad <> "const " <> p.packContainer <> " " <> raw <> " = " <> pointer <> "[" <> i' <> "];"]
        <> [pad <> "const " <> mslType t <> " " <> n <> " = " <> unpackElement p raw e <> ";" | (e, n) <- zip [0 ..] ns]
  SLoadPackVia ns t b e i -> do
    e' <- ex e
    i' <- ex i
    traverse_ (`declare` t) ns
    let p = packLayout t (length ns)
        raw = "fk_pack_" <> T.concat (take 1 ns)
        pointer = "((device const " <> p.packContainer <> "*)(" <> b <> "[" <> e' <> "]))"
    pure $
      [pad <> "const " <> p.packContainer <> " " <> raw <> " = " <> pointer <> "[" <> i' <> "];"]
        <> [pad <> "const " <> mslType t <> " " <> n <> " = " <> unpackElement p raw el <> ";" | (el, n) <- zip [0 ..] ns]
  SStorePack b i es -> do
    i' <- ex i
    es' <- traverse exNative es
    t <- gets (Map.findWithDefault TF32 b . (.envBuffers))
    let p = packLayout t (length es)
    c <- aliased p.packContainer
    pure [pad <> "reinterpret_cast<device " <> c <> "*>(" <> b <> ")[" <> i' <> "] = " <> packValue p es' <> ";"]
  STgStorePack a i es -> do
    i' <- ex i
    es' <- traverse exNative es
    t <- gets (Map.findWithDefault TF32 a . (.envArrays))
    -- Elements of four bytes go as their own vector, which the compiler
    -- keeps in registers and takes to alias them; narrower ones as words.
    let p = packLayout t (length es)
        v = mslType t <> showT (length es)
    (container, value) <-
      if scalarSize t == 4
        then pure (v, v <> "(" <> T.intercalate ", " es' <> ")")
        else (,packValue p es') <$> aliased p.packContainer
    pure [pad <> "reinterpret_cast<threadgroup " <> container <> "*>(" <> a <> ")[" <> i' <> "] = " <> value <> ";"]
  SIf _ c th el -> do
    c' <- ex c
    th' <- scoped (concat <$> traverse (stmt (depth + 1)) th)
    el' <- scoped (concat <$> traverse (stmt (depth + 1)) el)
    pure $
      [pad <> "if (" <> c' <> ") {"]
        <> th'
        <> (if null el then [] else [pad <> "} else {"] <> el')
        <> [pad <> "}"]
  SFor _ v start end step body -> do
    start' <- ex start
    end' <- ex end
    step' <- ex step
    body' <- scoped (declare v TU32 >> concat <$> traverse (stmt (depth + 1)) body)
    pure $
      [pad <> "for (uint " <> v <> " = " <> start' <> "; " <> v <> " < " <> end' <> "; " <> v <> " += " <> step' <> ") {"]
        <> body'
        <> [pad <> "}"]
  SBarrier -> pure [pad <> "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"]
  -- Pinned by the goldens alone: on the M3 Max, whose SIMD groups seem to
  -- run in lockstep, no run tells it from none.
  SSimdBarrier -> pure [pad <> "metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);"]
  SSimd n op t e -> do
    s <- line ("const " <> mslType t <> " " <> n <> " = " <> simdFn op <> "(") e ");"
    declare n t
    pure s
  SFragVar n t ->
    pure [pad <> fragment t <> " " <> n <> " = metal::make_filled_simdgroup_matrix<" <> mslType t <> ", 8, 8>(" <> literal (zero t) <> ");"]
  SFragLoad n m i st transposed -> do
    i' <- ex i
    st' <- ex st
    pure [pad <> "metal::simdgroup_load(" <> n <> ", " <> memory m <> " + " <> i' <> ", " <> st' <> (if transposed then ", ulong2(0, 0), true" else "") <> ");"]
  SFragMma d a b c -> pure [pad <> "metal::simdgroup_multiply_accumulate(" <> T.intercalate ", " [d, a, b, c] <> ");"]
  -- Apple's 8×8 layout: a lane holds two elements of one row, (lane / 4 &
  -- 4) + (lane / 2 % 4).
  SFragScaleRows f a i -> do
    i' <- ex i
    pure
      [ pad <> "{"
      , pad <> "    const float fk_scale = " <> a <> "[" <> i' <> " + (((fk_lane / 4u) & 4u) + ((fk_lane / 2u) % 4u))];"
      , pad <> "    " <> f <> ".thread_elements()[0] *= fk_scale;"
      , pad <> "    " <> f <> ".thread_elements()[1] *= fk_scale;"
      , pad <> "}"
      ]
  -- A lane's elements of the one, converted into the same of the other.
  SFragConvert d s -> pure [pad <> d <> ".thread_elements()[" <> e <> "] = static_cast<float>(" <> s <> ".thread_elements()[" <> e <> "]);" | e <- ["0", "1"]]
  SFragStore m i st n -> do
    i' <- ex i
    st' <- ex st
    pure [pad <> "metal::simdgroup_store(" <> n <> ", " <> memory m <> " + " <> i' <> ", " <> st' <> ");"]
  -- Refused before printing ('refusals').
  SStoreVia {} -> notPrinted
 where
  pad = T.replicate (4 * depth) " "
  fragment t = "metal::simdgroup_matrix<" <> mslType t <> ", 8, 8>"
  memory = \case
    FDevice b -> b
    FThreadgroup a -> a
    -- Refused before printing ('refusals').
    FVia {} -> notPrinted
  zero = \case
    TF16 -> LF16 0
    TBF16 -> LBF16 0
    _ -> LF32 0
  line prefix e suffix = do
    e' <- ex e
    pure [pad <> prefix <> e' <> suffix]
  -- A pack's container as its loads and stores name it: the typedef that
  -- may alias every type.
  aliased c = do
    modify' $ \s -> s {envHelpers = Set.insert (Container c) s.envHelpers}
    pure ("fk_" <> c)
  ex e = do
    env <- get
    modify' $ \s -> s {envHelpers = Set.union (helpersOf env e) s.envHelpers}
    pure (expr env e)
  -- A value as memory holds it: a bf16 held as its bits as a bfloat.
  exNative e = do
    e' <- ex e
    env <- get
    pure (if held env e then "as_type<bfloat>(" <> e' <> ")" else e')
  -- A bf16 value as its bits.
  exBits e = do
    e' <- ex e
    env <- get
    pure (if held env e then e' else "as_type<ushort>(" <> e' <> ")")
  -- Metal's own: simd_sum adds pairwise, neighbours first, as the IR orders
  -- a reduction (bit for bit on the M3 Max, in f32 and f16); simd_max and
  -- simd_min as metal::fmax and fmin.
  simdFn = \case
    SimdSum -> "metal::simd_sum"
    SimdMax -> "metal::simd_max"
    SimdMin -> "metal::simd_min"

------------------------------------------------------------------------------
-- Packs

-- | How a pack sits in memory: words of up to four bytes in one vector.
data PackLayout = PackLayout
  { packElement :: ScalarType
  , packContainer :: Text
  {- ^ @ushort@, @ushort2@, @uint@, @uint2@ or @uint4@, its loads and
  stores going through its may_alias name.
  -}
  , packWord :: Text
  -- ^ @ushort@ or @uint@.
  , packWords :: Int
  , packPerWord :: Int
  -- ^ Elements per word.
  }

-- | The gate made the size a power of two, from 2 to 16 bytes on AppleGPU9.
packLayout :: ScalarType -> Int -> PackLayout
packLayout t n =
  PackLayout
    { packElement = t
    , packContainer = if count == 1 then wordType else wordType <> showT count
    , packWord = wordType
    , packWords = count
    , packPerWord = wordBytes `div` size
    }
 where
  size = scalarSize t
  bytes = n * size
  -- Two 16-bit elements go as two words, a ushort2: before the
  -- containers' may_alias names, a vector was what Metal's compiler took
  -- to alias their half or bfloat accesses below MSL 4.1, a uint not.
  wordBytes = if size == 2 && bytes == 4 then 2 else min 4 bytes
  wordType = if wordBytes == 2 then "ushort" else "uint"
  count = bytes `div` wordBytes

-- | Element @e@ of a pack loaded into @raw@.
unpackElement :: PackLayout -> Text -> Int -> Text
unpackElement p raw e
  | p.packPerWord == 1 = "as_type<" <> mslType p.packElement <> ">(" <> w <> ")"
  | otherwise =
      "as_type<" <> mslType p.packElement <> showT p.packPerWord <> ">(" <> w <> ")." <> component (e `mod` p.packPerWord)
 where
  w = if p.packWords == 1 then raw else raw <> "." <> component (e `div` p.packPerWord)

-- | A pack's value from its elements.
packValue :: PackLayout -> [Text] -> Text
packValue p es = case map wordOf (chunks es) of
  [w] -> w
  ws -> p.packContainer <> "(" <> T.intercalate ", " ws <> ")"
 where
  wordOf = \case
    [e] | p.packPerWord == 1 -> "as_type<" <> p.packWord <> ">(" <> e <> ")"
    vs -> "as_type<" <> p.packWord <> ">(" <> mslType p.packElement <> showT p.packPerWord <> "(" <> T.intercalate ", " vs <> "))"
  chunks [] = []
  chunks xs = let (a, b) = splitAt p.packPerWord xs in a : chunks b

component :: Int -> Text
component = \case
  0 -> "x"
  1 -> "y"
  2 -> "z"
  _ -> "w"

------------------------------------------------------------------------------
-- Expressions

expr :: Env -> Expr -> Text
expr env = go
 where
  go = \case
    ELit (LBF16 b) -> "ushort(0x" <> T.pack (showHex b "") <> ")"
    ELit l -> literal l
    EVar n -> n
    EParam n -> "fk_p." <> n
    EBuiltin b -> case b of
      ThreadInGroup -> "fk_thread.x"
      GroupInGrid -> "fk_group.x"
      SimdLane -> "fk_lane"
      SimdGroup -> "fk_simd_group"
    EUn op e -> unary op (typeOf env e) (go e)
    EBin op a b
      | runTimeDivision env op a b -> (if op == Div then "fk_udiv(" else "fk_urem(") <> go a <> ", " <> go b <> ")"
      | otherwise -> binary op (typeOf env a) (go a) (go b)
    -- A half's on its bits: Metal's default options compile a select of a
    -- comparison as fmin.
    ESelect c a b
      | typeOf env a == TF16 -> "as_type<half>(metal::select(as_type<ushort>(" <> go b <> "), as_type<ushort>(" <> go a <> "), " <> go c <> "))"
      -- And a bfloat's, kept in its bits: Metal's compiler miscompiled a
      -- select of bfloat.
      | typeOf env a == TBF16 -> "metal::select(" <> bitsOf b <> ", " <> bitsOf a <> ", " <> go c <> ")"
      | otherwise -> "metal::select(" <> go b <> ", " <> go a <> ", " <> go c <> ")"
    EFma a b c -> "metal::fma(" <> go a <> ", " <> go b <> ", " <> go c <> ")"
    EConvert t e
      | typeOf env e == t -> go e
      -- A bf16 held as its bits widens by them.
      | typeOf env e == TBF16 && held env e -> widenBits t (go e)
      | t == TBF16 && byBits env e -> "fk_bf16(" <> (if typeOf env e == TF32 then go e else "static_cast<float>(" <> go e <> ")") <> ")"
      -- Through fk_keep: Metal's compiler narrowed a maximum, minimum or
      -- select of a constant toward zero, saturating.
      | t == TF16 && typeOf env e == TF32 -> "static_cast<half>(fk_keep(" <> go e <> "))"
      | otherwise -> "static_cast<" <> mslType t <> ">(" <> go e <> ")"
    ENarrowInt o t e -> narrowInt o t (go e)
    ELoad n i -> n <> "[" <> go i <> "]"
    ELoadVia n e i ->
      "((device const " <> mslType (Map.findWithDefault TF32 n env.envTables) <> "*)(" <> n <> "[" <> go e <> "]))[" <> go i <> "]"
    ETgLoad n i -> n <> "[" <> go i <> "]"
  bitsOf x = if held env x then go x else "as_type<ushort>(" <> go x <> ")"

{- | Whether a bf16 value is held as its bits, a @ushort@: a select's, a
literal's, a var's, a let's of one of those, and a narrowing's through
@fk_bf16@.
-}
held :: Env -> Expr -> Bool
held env e =
  typeOf env e == TBF16 && case e of
    ESelect {} -> True
    ELit _ -> True
    EVar n -> n `Set.member` env.envHeld
    EConvert _ x
      | typeOf env x == TBF16 -> held env x
      | otherwise -> byBits env x
    _ -> False

-- | bf16's bits widened: the upper half of a float, converted on if need be.
widenBits :: ScalarType -> Text -> Text
widenBits t u = if t == TF32 then f else "static_cast<" <> mslType t <> ">(" <> f <> ")"
 where
  f = "as_type<float>(uint(" <> u <> ") << 16)"

-- | Whether a narrowing to bf16 goes through @fk_bf16@: every one does.
byBits :: Env -> Expr -> Bool
byBits _ _ = True

{- | An integer narrowed: clamped to the narrower range, or its low bits,
an i32 through @uint@ and @as_type@ so that it wraps as the IR says.
-}
narrowInt :: Overflow -> ScalarType -> Text -> Text
narrowInt o t e = case (o, t) of
  (Clamp, TI8) -> "static_cast<char>(metal::clamp(" <> e <> ", -128, 127))"
  (Clamp, _) -> "static_cast<uchar>(metal::min(" <> e <> ", 255u))"
  (Wrap, TI8) -> "as_type<char>(static_cast<uchar>(as_type<uint>(" <> e <> ")))"
  (Wrap, _) -> "static_cast<uchar>(" <> e <> ")"

{- | A u32 quotient or remainder by a divisor not a literal: Metal's
compiler got such quotients of dividends near 2^32 wrong (4294967168 % 1
came out 4294967170), so they halve the dividend first.
-}
runTimeDivision :: Env -> BinOp -> Expr -> Expr -> Bool
runTimeDivision env op a b = op `elem` [Div, Rem] && typeOf env a == TU32 && not (isLiteral b)
 where
  isLiteral = \case
    ELit _ -> True
    _ -> False

-- | The helpers an expression's printing calls.
helpersOf :: Env -> Expr -> Set Helper
helpersOf env = go
 where
  go = \case
    EUn Abs x -> (if typeOf env x == TI32 then Set.insert AbsI32 else id) (go x)
    EUn op x | op `elem` [Sqrt, Rsqrt] && typeOf env x == TF32 -> Set.insert Keep (go x)
    EUn _ x -> go x
    EBin op a b -> (if runTimeDivision env op a b then Set.insert UDivU32 else id) (go a <> go b)
    ESelect c a b -> go c <> go a <> go b
    EFma a b c -> go a <> go b <> go c
    EConvert TBF16 x | typeOf env x /= TBF16 && byBits env x -> Set.insert NarrowBF16 (go x)
    EConvert TF16 x | typeOf env x == TF32 -> Set.insert Keep (go x)
    EConvert _ x -> go x
    ENarrowInt _ _ x -> go x
    ELoad _ i -> go i
    ELoadVia _ e i -> go e <> go i
    ETgLoad _ i -> go i
    ELit _ -> Set.empty
    EVar _ -> Set.empty
    EParam _ -> Set.empty
    EBuiltin _ -> Set.empty

unary :: UnOp -> ScalarType -> Text -> Text
unary op t e = case op of
  -- i32 through uint, so that it wraps as the IR defines it.
  Neg | t == TI32 -> "as_type<int>(0u - as_type<uint>(" <> e <> "))"
  Abs | t == TI32 -> "fk_abs_i32(" <> e <> ")"
  -- half through its bits, as IEEE 754's negate and abs: Metal's own give
  -- a NaN the canonical one.
  Neg | t == TF16 -> "as_type<half>(ushort(as_type<ushort>(" <> e <> ") ^ 0x8000u))"
  Abs | t == TF16 -> "as_type<half>(ushort(as_type<ushort>(" <> e <> ") & 0x7fffu))"
  -- float's magnitude through its bits too: macOS 27.2's compiler drops
  -- metal::abs where it takes the operand for non-negative, so that
  -- abs(rsqrt(-0)) came out -inf. Its negation it keeps.
  Abs | t == TF32 -> "as_type<float>(as_type<uint>(" <> e <> ") & 0x7fffffffu)"
  Neg -> "(-" <> e <> ")"
  Not -> "(!" <> e <> ")"
  Abs -> "metal::abs(" <> e <> ")"
  _ | t == TF16 -> "static_cast<half>(" <> precise <> "(static_cast<float>(" <> e <> ")))"
  -- Through fk_keep: Metal's compiler didn't round the sqrt or rsqrt of a
  -- widened half or bfloat correctly.
  _ | op `elem` [Sqrt, Rsqrt] && t == TF32 -> precise <> "(fk_keep(" <> e <> "))"
  _ -> precise <> "(" <> e <> ")"
 where
  precise = case op of
    Sqrt -> "metal::precise::sqrt"
    Rsqrt -> "metal::precise::rsqrt"
    Exp -> "metal::precise::exp"
    Log -> "metal::precise::log"
    Tanh -> "metal::precise::tanh"
    _ -> ""

binary :: BinOp -> ScalarType -> Text -> Text -> Text
binary op t a b = case op of
  -- Metal's own, as the IR allows: a NaN operand gives way. Which of
  -- operands that compare equal (-0 and +0) or of two NaNs comes out is
  -- the target's: MSL (section 6.6) gives the first of equal operands, a
  -- call alone in a kernel gave the second on the M3 Max, and the
  -- compiler took fmin(a, b) and fmin(b, a) in one kernel for one call.
  -- binary32's precise ones, whatever the compile options; binary16 has no
  -- precise overloads, and the pragma and the required options hold its.
  Min | t == TF32 -> "metal::precise::fmin(" <> a <> ", " <> b <> ")"
  Max | t == TF32 -> "metal::precise::fmax(" <> a <> ", " <> b <> ")"
  Min | floating -> "metal::fmin(" <> a <> ", " <> b <> ")"
  Max | floating -> "metal::fmax(" <> a <> ", " <> b <> ")"
  Min -> "metal::min(" <> a <> ", " <> b <> ")"
  Max -> "metal::max(" <> a <> ", " <> b <> ")"
  -- Both operands doubled first, exactly: Metal's compiler rounded a
  -- constant divisor's quotient wrongly (x / 3 1 ulp off for a third of
  -- the halves).
  Div | t == TF16 -> "static_cast<half>((static_cast<float>(" <> a <> ") * 2.0f) / (static_cast<float>(" <> b <> ") * 2.0f))"
  -- i32 through uint, so that it wraps as the IR defines it.
  _ | t == TI32 && op `elem` [Add, Sub, Mul] -> "as_type<int>(as_type<uint>(" <> a <> ") " <> symbol <> " as_type<uint>(" <> b <> "))"
  -- Both operands evaluated, as the IR says.
  And -> "bool(int(" <> a <> ") & int(" <> b <> "))"
  Or -> "bool(int(" <> a <> ") | int(" <> b <> "))"
  _ -> "(" <> a <> " " <> symbol <> " " <> b <> ")"
 where
  floating = t `elem` [TF32, TF16]
  symbol = case op of
    Add -> "+"
    Sub -> "-"
    Mul -> "*"
    Div -> "/"
    Rem -> "%"
    Eq -> "=="
    Ne -> "!="
    Lt -> "<"
    Le -> "<="
    Gt -> ">"
    Ge -> ">="
    And -> "&"
    Or -> "|"
    _ -> "?"

literal :: Literal -> Text
literal = \case
  LF32 x
    | isNaN x || isInfinite x -> "as_type<float>(" <> hexU32 (castFloatToWord32 x) <> ")"
    | x < 0 || isNegativeZero x -> "(" <> T.pack (show x) <> "f)"
    | otherwise -> T.pack (show x) <> "f"
  LF16 b -> "as_type<half>(ushort(" <> hexU16 b <> "))"
  LBF16 b -> "as_type<bfloat>(ushort(" <> hexU16 b <> "))"
  LI32 n
    | n == minBound -> "(-2147483647 - 1)"
    | n < 0 -> "(" <> showT n <> ")"
    | otherwise -> showT n
  LU32 n -> showT n <> "u"
  LI8 n -> "char(" <> showT n <> ")"
  LU8 n -> "uchar(" <> showT n <> ")"
  LBool b -> if b then "true" else "false"
 where
  hexU32 w = "0x" <> T.pack (showHex w "") <> "u"
  hexU16 w = "0x" <> T.pack (showHex w "")

mslType :: ScalarType -> Text
mslType = \case
  TF32 -> "float"
  TF16 -> "half"
  TBF16 -> "bfloat"
  TI32 -> "int"
  TU32 -> "uint"
  TI8 -> "char"
  TU8 -> "uchar"
  TBool -> "bool"

showT :: (Show a) => a -> Text
showT = T.pack . show
