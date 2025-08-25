{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "dev-env";

  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.numpy
    pkgs.python312Packages.scikit-learn
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.seaborn
    pkgs.python312Packages.torch
    pkgs.python312Packages.pylsp-rope
    pkgs.ruff

    pkgs.gcc         
    pkgs.stdenv.cc  
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/
  '';
}

