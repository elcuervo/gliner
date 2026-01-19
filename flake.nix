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
        lib = pkgs.lib;

        isLinux = pkgs.stdenv.isLinux;

        python-with-packages = pkgs.python3.withPackages (ps: with ps; [
          huggingface-hub
        ]);

        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          openssl
        ];

        rubyWrapped = pkgs.writeShellScriptBin "ruby" ''
          ${lib.optionalString isLinux ''export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH"''}
          exec ${pkgs.ruby}/bin/ruby "$@"
        '';

        bundleWrapped = pkgs.writeShellScriptBin "bundle" ''
          ${lib.optionalString isLinux ''export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH"''}
          exec ${pkgs.bundler}/bin/bundle "$@"
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            [
              rubyWrapped
              bundleWrapped

              pkgs.ruby
              pkgs.bundler
              pkgs.rubocop

              python-with-packages
              pkgs.pipenv
              pkgs.black
            ]
            ++ lib.optionals isLinux ([ pkgs.nix-ld ] ++ runtimeLibs);

          NIX_LD = lib.optionalString isLinux "${pkgs.nix-ld}/libexec/nix-ld";
          NIX_LD_LIBRARY_PATH = lib.optionalString isLinux (lib.makeLibraryPath runtimeLibs);

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
