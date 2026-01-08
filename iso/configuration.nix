{ pkgs, lib, config, ... }: {
  networking.hostName = "hydra-live";
  
  # --- Hybrid Driver Support ---
  nixpkgs.config.allowUnfree = true;
  
  # Load kernel modules for both
  boot.initrd.kernelModules = [ "amdgpu" "nvidia" "nvidia_modeset" "nvidia_uvm" "nvidia_drm" ];
  
  # XServer Drivers: Try NVIDIA first, then AMD
  services.xserver.videoDrivers = [ "nvidia" "amdgpu" ];
  
  # OpenGL / OpenCL / Vulkan
  hardware.opengl = {
    enable = true;
    driSupport = true;
    driSupport32Bit = true;
    extraPackages = with pkgs; [
      # ROCm OpenCL & Runtime
      rocmPackages.clr.icd 
      rocmPackages.rocminfo
      amdvlk
    ];
  };

  # NVIDIA Specifics
  hardware.nvidia.open = false;

  # --- System Optimization ---
  boot.kernelParams = [ "copytoram" ]; 
  services.openssh.enable = true;
  users.users.nixos.password = "hydra"; 

  # --- HydraMesh Defaults ---
  # To enable AMD ROCm worker on boot, change backend to "rocm"
  services.hydramesh = {
    enable = false;
    role = "worker";
    backend = "nvidia"; # Change to "rocm" for AMD cards
  };

  environment.systemPackages = with pkgs; [
    tmux ripgrep fd bat 
    pciutils # lspci to check GPU
    rocmPackages.rocminfo # verify AMD
  ];
}
