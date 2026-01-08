{
  description = "HydraMesh DCF: Containers & All-in-One ISO";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # Binary Cache (Critical for CUDA/Torch)
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, nixos-generators, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config = { allowUnfree = true; cudaSupport = true; };
        };

        # --- Python Environments ---
        # Lightweight env for Router/Head
        pythonCore = pkgs.python311.withPackages (ps: with ps; [
          pydantic requests numpy
        ]);
        
        # Heavy env for Worker (ML Stack)
        pythonML = pkgs.python311.withPackages (ps: with ps; [
          torch accelerate safetensors huggingface-hub
          transformers diffusers bitsandbytes sentencepiece protobuf
          pydantic requests numpy
        ]);

        # Rust Toolchain (for DevShell/ISO)
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };

        # Source Code Derivation (Keeps containers clean)
        meshSource = pkgs.stdenv.mkDerivation {
          name = "hydramesh-src";
          src = ./src;
          installPhase = ''
            mkdir -p $out/bin
            cp *.py $out/bin/
            chmod +x $out/bin/*.py
          '';
        };

      in {
        # ==========================================
        # 1. Individual Containers (Head & Worker)
        # ==========================================
        packages.container-head = pkgs.dockerTools.buildLayeredImage {
          name = "demod/mesh-head";
          tag = "latest";
          contents = [ pythonCore meshSource pkgs.bash pkgs.coreutils ];
          config = {
            Cmd = [ "python3" "${meshSource}/bin/head_controller.py" ];
            Env = [ "PYTHONPATH=${pythonCore}/${pythonCore.sitePackages}" ];
            ExposedPorts = { "7777/udp" = {}; "7778/udp" = {}; };
            WorkingDir = "/var/lib/hydramesh";
          };
        };

        packages.container-worker = pkgs.dockerTools.buildLayeredImage {
          name = "demod/mesh-worker";
          tag = "latest";
          # Maximize layering for huge ML libs
          maxLayers = 120;
          contents = [ pythonML meshSource pkgs.cudaPackages.cudatoolkit pkgs.bash ];
          config = {
            Cmd = [ "python3" "${meshSource}/bin/worker_node.py" ];
            Env = [
              "PYTHONPATH=${pythonML}/${pythonML.sitePackages}"
              "LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib"
              "HF_HOME=/data/huggingface"
            ];
            ExposedPorts = { "7779/udp" = {}; };
            WorkingDir = "/data";
            Volumes = { "/data" = {}; };
          };
        };

        # ==========================================
        # 2. All-in-One ISO (Host + Dev Environment)
        # ==========================================
        packages.iso = nixos-generators.nixosGenerate {
          inherit system;
          modules = [
            self.nixosModules.default # Include our service definitions
            ./iso/configuration.nix   # ISO-specific config
            ({ pkgs, ... }: {
               # Add the entire Dev Toolchain to the ISO
               environment.systemPackages = [
                 pythonML
                 rustToolchain
                 pkgs.git
                 pkgs.htop
                 pkgs.nvtopPackages.full
                 pkgs.neovim
                 # Include the local source scripts in the path
                 meshSource
               ];
            })
          ];
          format = "install-iso";
        };

        # ==========================================
        # 3. DevShell (Local Development)
        # ==========================================
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonML rustToolchain pkgs.cargo-watch pkgs.cudaPackages.cudatoolkit ];
          shellHook = ''
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
            export RUST_LOG=info
            echo "HydraMesh Dev Environment Loaded."
          '';
        };

        # ==========================================
        # 4. NixOS Module (Service Definition)
        # ==========================================
        nixosModules.default = { config, lib, pkgs, ... }: 
          let cfg = config.services.hydramesh; in {
            options.services.hydramesh = {
              enable = lib.mkEnableOption "HydraMesh DCF Service";
              role = lib.mkOption { type = lib.types.enum [ "head" "worker" ]; default = "worker"; };
              headIp = lib.mkOption { type = lib.types.str; default = "127.0.0.1"; };
              hfToken = lib.mkOption { type = lib.types.str; default = ""; };
            };

            config = lib.mkIf cfg.enable {
              users.users.hydramesh = { isSystemUser = true; group = "hydramesh"; home = "/var/lib/hydramesh"; createHome = true; };
              users.groups.hydramesh = {};
              networking.firewall.allowedUDPPorts = [ 7777 7778 7779 9999 ];

              systemd.services.hydramesh = {
                description = "HydraMesh Node (${cfg.role})";
                wantedBy = [ "multi-user.target" ];
                after = [ "network.target" ];
                environment.HF_TOKEN = cfg.hfToken;
                serviceConfig = {
                  User = "hydramesh";
                  Group = "hydramesh";
                  WorkingDirectory = "/var/lib/hydramesh";
                  Restart = "always";
                  ExecStart = if cfg.role == "head" then
                    "python3 ${meshSource}/bin/head_controller.py"
                  else
                    "python3 ${meshSource}/bin/worker_node.py --head-ip ${cfg.headIp}";
                };
              };
            };
        };
      }
    );
}
