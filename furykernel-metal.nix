{ mkDerivation, base, containers, furykernel, lib, tasty
, tasty-hunit, text, transformers
}:
mkDerivation {
  pname = "furykernel-metal";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [
    base containers furykernel text transformers
  ];
  testHaskellDepends = [ base furykernel tasty tasty-hunit text ];
  homepage = "https://github.com/infurer/furykernel-metal";
  description = "The Apple Metal target of FuryKernel";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
