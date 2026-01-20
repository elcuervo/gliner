{
  description = "gliner2 Ruby development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs-ruby = {
      url = "github:bobvanderlinden/nixpkgs-ruby";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nixpkgs-ruby }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        isLinux = pkgs.stdenv.isLinux;

        rubyPkg = nixpkgs-ruby.lib.packageFromRubyVersionFile {
          file = ./.ruby-version;
          inherit system;
        };

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
          exec ${rubyPkg}/bin/ruby "$@"
        '';

        bundleWrapped = pkgs.writeShellScriptBin "bundle" ''
          ${lib.optionalString isLinux ''export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH"''}
          exec ${rubyPkg}/bin/bundle "$@"
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            [
              rubyWrapped
              bundleWrapped

              rubyPkg

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
            export PATH="${rubyPkg}/bin:$PATH"
            export PATH="$PWD/.bundle/bin:$PATH"
          '';
        };
      });
}
