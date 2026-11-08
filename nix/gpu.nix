# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# `nix run .#gpu-conformance`: the GPU conformance run, outside CI, on an
# Apple9 Mac. test/gpu/Cases.hs writes the corpus (gpu-corpus): the core's
# checks.cpu kernels, the core's test kernels and this library's own
# constructs, each lowered by `metal`, with the interpreter's inputs and
# outputs. The runner (gpu-runner, test/gpu) compiles each with the
# lowered options, launches it as FuryKernel.Metal tells a runtime to,
# and holds its outputs to the interpreter's by the IR's leave: as a
# runtime runs it, under shader validation, and compiled with Metal's
# default options. The run prints the GPU, macOS and the compiler. The
# corpus is a check on every system too, `checks.gpu-corpus`: its
# generator, compiled under -Wall -Werror, needs only GHC.
{
  inputs,
  self,
  ...
}: {
  perSystem = {
    config,
    craneLib,
    gpuArgs,
    gpuArtifacts,
    haskellPackages,
    lib,
    pkgs,
    system,
    ...
  }:
    lib.mkMerge [
      {
        packages.gpu-corpus = let
          ghc = haskellPackages.ghcWithPackages (_: [config.packages.furykernel-metal]);
        in
          pkgs.runCommand "furykernel-metal-gpu-corpus" {
            nativeBuildInputs = [ghc];
          } ''
            ghc -O -Wall -Werror -XGHC2024 -XOverloadedRecordDot -XOverloadedStrings -XStrictData -XTypeFamilies \
              -i${inputs.furykernel}/test -outputdir "$TMPDIR/build" -o "$TMPDIR/cases" ${self}/test/gpu/Cases.hs
            mkdir -p $out
            "$TMPDIR/cases" $out
          '';
        checks.gpu-corpus = config.packages.gpu-corpus;
      }
      (lib.mkIf (system == "aarch64-darwin") {
        packages.gpu-runner = craneLib.buildPackage (gpuArgs
          // {
            cargoArtifacts = gpuArtifacts;
            doCheck = false;
          });
        checks.gpu-runner-clippy = craneLib.cargoClippy (gpuArgs
          // {
            cargoArtifacts = gpuArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });
        # The runner's comparison, which needs no GPU: its unit tests.
        checks.gpu-runner-tests = craneLib.cargoTest (gpuArgs
          // {
            cargoArtifacts = gpuArtifacts;
          });
        apps.gpu-conformance = {
          type = "app";
          program = lib.getExe (pkgs.writeShellApplication {
            name = "gpu-conformance";
            text = ''
              echo "compiler: GPUCompiler $(readlink /System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/Current)"
              # First as a runtime runs it, every check made. Then under shader
              # validation, whose reports of invalid loads and stores, in the
              # command buffers' logs, fail a case (and go to stderr too), and
              # which instruments threadgroup memory, so that its size isn't
              # compared there. Last with Metal's default options, fast math
              # among them, as a runtime that ignores the lowered options
              # compiles: the source's pragmas and precise functions must hold.
              status=0
              runner=${config.packages.gpu-runner}/bin/furykernel-metal-gpu-conformance
              "$runner" ${config.packages.gpu-corpus} || status=1
              MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 "$runner" ${config.packages.gpu-corpus} || status=1
              "$runner" --default-options ${config.packages.gpu-corpus} || status=1
              exit "$status"
            '';
          });
          meta.description = "Run every case of the GPU conformance corpus on this Mac's GPU";
        };
      })
    ];
}
