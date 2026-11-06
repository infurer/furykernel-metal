# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# The Rust toolchain from rust-overlay (test/gpu/rust-toolchain.toml) and
# crane, for the GPU conformance runner, and rustfmt for `nix fmt`. One
# buildDepsOnly produces the runner's dependencies' cargoArtifacts, which
# its build and its clippy reuse; they are built on aarch64-darwin only
# (nix/gpu.nix), since the runner links Metal.
{inputs, ...}: {
  perSystem = {
    lib,
    system,
    ...
  }: let
    pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [inputs.rust-overlay.overlays.default];
    };
    rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../test/gpu/rust-toolchain.toml;
    craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
    gpuArgs = {
      pname = "furykernel-metal-gpu-conformance";
      version = "0.0.0";
      strictDeps = true;
      src = lib.fileset.toSource {
        root = ../test/gpu;
        fileset = lib.fileset.unions [../test/gpu/Cargo.toml ../test/gpu/Cargo.lock ../test/gpu/src];
      };
      cargoVendorDir = craneLib.vendorCargoDeps {cargoLock = ../test/gpu/Cargo.lock;};
    };
  in {
    _module.args = {
      inherit craneLib gpuArgs rustToolchain;
      gpuArtifacts = craneLib.buildDepsOnly gpuArgs;
    };
  };
}
