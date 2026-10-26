{ mkDerivation, base, furykernel, lib, tasty, tasty-hunit, text }:
mkDerivation {
  pname = "furykernel-metal";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [ base furykernel text ];
  testHaskellDepends = [ base furykernel tasty tasty-hunit ];
  homepage = "https://github.com/infurer/furykernel-metal";
  description = "The Apple Metal target of FuryKernel";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
