{
  description = "Safe Tensor Inference Environment (Image & Text) with CUDA Support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # Binary cache configuration to avoid compiling PyTorch/CUDA from source
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

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        # Python environment with critical ML libraries
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Core
          torch
          numpy
          accelerate
          safetensors
          huggingface-hub
          scipy
          
          # Image Generation
          torchvision
          transformers
          diffusers
          pillow
          xformers
          opencv4
          
          # Text Generation (LLMs)
          sentencepiece
          bitsandbytes  # Required for 4-bit quantization
          protobuf
          
          # UI
          gradio
        ]);

        # Dynamic environment wrapper to handle permissions for both Service and User modes
        envWrapper = ''
          if [ "$USER" = "inference-service" ]; then
             export BASE_DIR="/var/lib/inference-models"
          else
             export BASE_DIR="$HOME/.cache/inference-models"
             mkdir -p "$BASE_DIR"
          fi
          
          export HF_HOME="$BASE_DIR/huggingface"
          export XDG_CACHE_HOME="$BASE_DIR/cache"
          export PYTHONPATH=${pythonEnv}/lib/python3.11/site-packages:$PYTHONPATH
        '';

      in {
        packages = {
          # Image Inference CLI
          image-inference = pkgs.writeShellApplication {
            name = "image-inference";
            runtimeInputs = [ pythonEnv pkgs.cudaPackages.cudatoolkit ];
            text = ''
              ${envWrapper}
              python ${./inference_image.py} "$@"
            '';
          };

          # Text Inference CLI
          text-inference = pkgs.writeShellApplication {
            name = "text-inference";
            runtimeInputs = [ pythonEnv pkgs.cudaPackages.cudatoolkit ];
            text = ''
              ${envWrapper}
              python ${./inference_text.py} "$@"
            '';
          };

          # Web UI (Unified)
          webui = pkgs.writeShellApplication {
            name = "inference-webui";
            runtimeInputs = [ pythonEnv pkgs.cudaPackages.cudatoolkit ];
            text = ''
              ${envWrapper}
              echo "Starting Web UI on port 7860..."
              echo "Models will be stored in: $BASE_DIR"
              python -m gradio ${./webui.py}
            '';
          };

          default = self.packages.${system}.webui;
        };

        apps = {
          image = flake-utils.lib.mkApp { drv = self.packages.${system}.image-inference; };
          text = flake-utils.lib.mkApp { drv = self.packages.${system}.text-inference; };
          webui = flake-utils.lib.mkApp { drv = self.packages.${system}.webui; };
          default = self.apps.${system}.webui;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pythonEnv
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
            git
            git-lfs
            wget
            curl
            jq
            htop
            nvtopPackages.full
          ];

          shellHook = ''
            ${envWrapper}
            export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
            
            echo "==========================================="
            echo "  AI Inference Dev Environment Loaded      "
            echo "==========================================="
            echo "Models dir: $BASE_DIR"
            echo "CUDA Check: $(python -c 'import torch; print(torch.cuda.is_available())')"
            echo "Commands:"
            echo "  nix run .#image -- [args]"
            echo "  nix run .#text -- [args]"
            echo "  nix run .#webui"
          '';
        };

        # NixOS Module for systemd deployment
        nixosModules.default = { config, lib, pkgs, ... }: 
          let
            cfg = config.services.inference-server;
          in {
            options.services.inference-server = {
              enable = lib.mkEnableOption "AI Inference Service";
              
              imageModel = lib.mkOption {
                type = lib.types.str;
                default = "runwayml/stable-diffusion-v1-5";
                description = "Default Stable Diffusion model ID";
              };

              textModel = lib.mkOption {
                type = lib.types.str;
                default = "mistralai/Mistral-7B-v0.1";
                description = "Default LLM model ID";
              };
              
              port = lib.mkOption {
                type = lib.types.port;
                default = 7860;
              };
            };

            config = lib.mkIf cfg.enable {
              # Create the system user
              users.users.inference-service = {
                isSystemUser = true;
                group = "inference-service";
                home = "/var/lib/inference-models";
                createHome = true;
              };
              users.groups.inference-service = {};

              # Allow the port
              networking.firewall.allowedTCPPorts = [ cfg.port ];

              # Systemd Service
              systemd.services.inference-server = {
                description = "AI Inference Web UI Service";
                wantedBy = [ "multi-user.target" ];
                after = [ "network.target" ];
                
                serviceConfig = {
                  Type = "simple";
                  User = "inference-service";
                  Group = "inference-service";
                  WorkingDirectory = "/var/lib/inference-models";
                  ExecStart = "${self.packages.${system}.webui}/bin/inference-webui";
                  Restart = "on-failure";
                  Environment = [
                    "SD_MODEL_ID=${cfg.imageModel}"
                    "LLM_MODEL_ID=${cfg.textModel}"
                  ];
                };
              };
            };
        };
      }
    );
}
