{ pkgs, lib, config, ... }: {
  # --- 1. System Identity ---
  networking.hostName = "hydra-dev-live";
  
  # --- 2. Hardware Support ---
  # Enable Proprietary NVIDIA Drivers for GPU Inference
  nixpkgs.config.allowUnfree = true;
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.opengl.enable = true;
  hardware.nvidia.open = false; # Use proprietary for max CUDA compat

  # --- 3. Live System Optimization ---
  # Load the OS into RAM for performance (requires >8GB RAM)
  boot.kernelParams = [ "copytoram" ]; 
  
  # Enable SSH so you can remote into this dev box
  services.openssh.enable = true;
  users.users.nixos.password = "hydra"; 

  # --- 4. HydraMesh Service (Disabled by default) ---
  # The module is loaded (via flake.nix), but disabled.
  # To start it as a worker: 
  #   sudo systemctl start hydramesh
  # To configure it dynamically:
  #   Edit /etc/nixos/configuration.nix or set flags at runtime
  services.hydramesh.enable = false; 

  # --- 5. Dev Experience ---
  # Nice-to-haves for a terminal dev environment
  programs.zsh.enable = true;
  users.defaultUserShell = pkgs.zsh;
  environment.systemPackages = with pkgs; [
    tmux
    ripgrep
    fd
    bat
  ];
}
