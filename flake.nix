{
  description = "HydraMesh DCF: Hybrid CUDA/ROCm Cluster";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
      "https://rocm.cachix.org" 
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "rocm.cachix.org-1:6QLNg1NbVzc7Q87rK0Kqa+4L5+jO2r+q7Q+O9X+vI4Q="
    ];
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, nixos-generators, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # --- 1. Define Package Sets for Each Vendor ---
        
        # NVIDIA Version (CUDA Enabled)
        pkgsCuda = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config = { allowUnfree = true; cudaSupport = true; };
        };

        # AMD Version (ROCm Enabled, CUDA Disabled)
        pkgsRocm = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config = { allowUnfree = true; rocmSupport = true; cudaSupport = false; };
        };

        # Rust Toolchain (Shared)
        rustToolchain = pkgsCuda.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };

        # Source Code (Shared)
        meshSource = pkgsCuda.stdenv.mkDerivation {
          name = "hydramesh-src";
          src = ./src;
          installPhase = ''
            mkdir -p $out/bin
            cp *.py $out/bin/
            chmod +x $out/bin/*.py
          '';
        };

        # --- 2. Python Environments ---

        # Common Packages
        commonPy = ps: with ps; [ 
          pydantic requests numpy pillow omegaconf protobuf 
        ];

        # NVIDIA Python Env
        pythonML_Nvidia = pkgsCuda.python311.withPackages (ps: (commonPy ps) ++ (with ps; [
          torch accelerate safetensors huggingface-hub
          transformers diffusers bitsandbytes sentencepiece
        ]));

        # AMD Python Env
        # Note: bitsandbytes often has issues on ROCm without specific patches. 
        # We assume standard torch support here.
        pythonML_Rocm = pkgsRocm.python311.withPackages (ps: (commonPy ps) ++ (with ps; [
          torch accelerate safetensors huggingface-hub
          transformers diffusers sentencepiece
        ]));

      in {
        packages = {
          # --- Containers ---

          # 1. Head Node (Vendor Neutral)
          container-head = pkgsCuda.dockerTools.buildLayeredImage {
            name = "alh477/mesh-head";
            tag = "latest";
            contents = [ pythonML_Nvidia meshSource pkgsCuda.bash pkgsCuda.coreutils ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/head_controller.py" ];
              Env = [ "PYTHONPATH=${pythonML_Nvidia}/${pythonML_Nvidia.sitePackages}" ];
              ExposedPorts = { "7777/udp" = {}; "7778/udp" = {}; };
              WorkingDir = "/var/lib/hydramesh";
            };
          };

          # 2. Worker Node (NVIDIA)
          container-worker-nvidia = pkgsCuda.dockerTools.buildLayeredImage {
            name = "alh477/mesh-worker-nvidia";
            tag = "latest";
            maxLayers = 120;
            contents = [ pythonML_Nvidia meshSource pkgsCuda.cudaPackages.cudatoolkit pkgsCuda.bash ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/worker_node.py" ];
              Env = [
                "PYTHONPATH=${pythonML_Nvidia}/${pythonML_Nvidia.sitePackages}"
                "LD_LIBRARY_PATH=${pkgsCuda.cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib"
                "HF_HOME=/data/huggingface"
              ];
              ExposedPorts = { "7779/udp" = {}; };
              WorkingDir = "/data";
              Volumes = { "/data" = {}; "/models" = {}; };
            };
          };

          # 3. Worker Node (AMD ROCm)
          container-worker-rocm = pkgsRocm.dockerTools.buildLayeredImage {
            name = "demod/mesh-worker-rocm";
            tag = "latest";
            maxLayers = 120;
            contents = [ 
              pythonML_Rocm 
              meshSource 
              pkgsRocm.rocmPackages.clr 
              pkgsRocm.rocmPackages.rocminfo 
              pkgsRocm.bash 
            ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/worker_node.py" ];
              Env = [
                "PYTHONPATH=${pythonML_Rocm}/${pythonML_Rocm.sitePackages}"
                # ROCm Libraries path
                "LD_LIBRARY_PATH=${pkgsRocm.rocmPackages.clr}/lib:/run/opengl-driver/lib"
                "HSA_OVERRIDE_GFX_VERSION=10.3.0" # Optional: Override for specific cards (e.g. 6700XT)
                "HF_HOME=/data/huggingface"
              ];
              ExposedPorts = { "7779/udp" = {}; };
              WorkingDir = "/data";
              Volumes = { "/data" = {}; "/models" = {}; };
            };
          };

          # --- All-in-One ISO ---
          iso = nixos-generators.nixosGenerate {
            inherit system;
            modules = [
              self.nixosModules.default 
              ./iso/configuration.nix   
              ({ pkgs, ... }: {
                 environment.systemPackages = [
                   # Default to NVIDIA env for the host shell, 
                   # but allow ROCm libs to be present
                   pythonML_Nvidia 
                   rustToolchain
                   pkgs.git pkgs.htop pkgs.nvtopPackages.full
                   pkgs.rocmPackages.rocminfo # AMD Tool
                   pkgs.rocmPackages.rocm-smi # AMD Monitor
                   meshSource
                 ];
              })
            ];
            format = "install-iso";
          };
        };

        # --- Dev Shells ---
        
        # Default (NVIDIA)
        devShells.default = pkgsCuda.mkShell {
          buildInputs = [ pythonML_Nvidia rustToolchain pkgsCuda.cudaPackages.cudatoolkit ];
          shellHook = ''
             export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
             echo "HydraMesh Dev (NVIDIA) Loaded"
          '';
        };

        # AMD ROCm Shell (`nix develop .#rocm`)
        devShells.rocm = pkgsRocm.mkShell {
          buildInputs = [ pythonML_Rocm rustToolchain pkgsRocm.rocmPackages.clr ];
          shellHook = ''
             export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgsRocm.rocmPackages.clr}/lib:$LD_LIBRARY_PATH
             export HSA_OVERRIDE_GFX_VERSION=10.3.0
             echo "HydraMesh Dev (ROCm) Loaded"
          '';
        };

        # --- NixOS Module ---
        nixosModules.default = { config, lib, pkgs, ... }: 
          let cfg = config.services.hydramesh; in {
            options.services.hydramesh = {
              enable = lib.mkEnableOption "HydraMesh DCF Service";
              role = lib.mkOption { type = lib.types.enum [ "head" "worker" ]; default = "worker"; };
              # New Option: Hardware Backend
              backend = lib.mkOption { type = lib.types.enum [ "nvidia" "rocm" ]; default = "nvidia"; };
              headIp = lib.mkOption { type = lib.types.str; default = "127.0.0.1"; };
              hfToken = lib.mkOption { type = lib.types.str; default = ""; };
              localModelPath = lib.mkOption { type = lib.types.str; default = ""; };
            };

            config = lib.mkIf cfg.enable {
              users.users.hydramesh = { isSystemUser = true; group = "hydramesh"; home = "/var/lib/hydramesh"; createHome = true; };
              users.groups.hydramesh = {};
              networking.firewall.allowedUDPPorts = [ 7777 7778 7779 9999 ];

              systemd.services.hydramesh = {
                description = "HydraMesh Node (${cfg.role} - ${cfg.backend})";
                wantedBy = [ "multi-user.target" ];
                after = [ "network.target" ];
                environment = {
                  HF_TOKEN = cfg.hfToken;
                  # Inject ROCm override if needed
                  HSA_OVERRIDE_GFX_VERSION = if cfg.backend == "rocm" then "10.3.0" else "";
                };
                serviceConfig = {
                  User = "hydramesh";
                  Group = "hydramesh";
                  WorkingDirectory = "/var/lib/hydramesh";
                  Restart = "always";
                  OOMScoreAdjust = -500;
                  
                  # Select the correct binary based on backend config
                  ExecStart = 
                    let 
                      pEnv = if cfg.backend == "rocm" then pythonML_Rocm else pythonML_Nvidia;
                    in
                    if cfg.role == "head" then
                      "${pkgsCuda.python311}/bin/python3 ${meshSource}/bin/head_controller.py"
                    else
                      "${pEnv}/bin/python3 ${meshSource}/bin/worker_node.py --head-ip ${cfg.headIp} ${if cfg.localModelPath != "" then "--model-path " + cfg.localModelPath else ""}";
                };
              };
            };
        };
      }
    );
}
