{
  description = "JUMP production pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            nvidia.acceptLicense = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pixi
          ];

          shellHook = ''
            echo "jpx - JUMP Production eXplore"
            echo ""
            echo "Usage:"
            echo "  pixi install          # Install dependencies"
            echo "  just redun main       # Run full pipeline"
            echo "  just --list           # Show available commands"
          '';

          # Use system NVIDIA driver libraries to avoid version mismatch
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
        };
      });
}
