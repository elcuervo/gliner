{
  description = "gliner2 Ruby development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ruby
            bundler
            rubocop
            python3
            pipenv
            black
          ];

          BUNDLE_PATH = ".bundle";
          BUNDLE_APP_CONFIG = ".bundle";
          BUNDLE_BIN = ".bundle/bin";
          BUNDLE_DISABLE_SHARED_GEMS = "true";

          shellHook = ''
            export PATH="$PWD/.bundle/bin:$PATH"
          '';
        };
      });
}
