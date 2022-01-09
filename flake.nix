{
  description = "An over-engineered Hello World in bash";

  # Nixpkgs / NixOS version to use.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs
            { inherit system; config.allowUnfree = true; };
          lib = pkgs.lib;
        in
        {
          devShell =
            pkgs.stdenv.mkDerivation {
              name = "cuda-env-shell";
              buildInputs = with pkgs; [
                git
                gitRepo
                gnupg
                autoconf
                curl
                procps
                gnumake
                utillinux
                m4
                gperf
                unzip
                cudaPackages.cudatoolkit_11_0
                linuxPackages.nvidia_x11
                libGLU
                libGL
                xorg.libXi
                xorg.libXmu
                freeglut
                xorg.libXext
                xorg.libX11
                xorg.libXv
                xorg.libXrandr
                zlib
                ncurses5
                stdenv.cc
                binutils
                # These libraries are requires for OpenGL
                xorg.libX11.dev
                xlibs.xorgproto
                freeglut
                libGLU.dev
                libGL.dev
                libGL_driver
                # Nvidia Nsight
                jdk8
                # Cmake
                cmake
              ];
              shellHook = ''
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit_11_0}
                export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
                export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
                export EXTRA_CCFLAGS="-I/usr/include"

                # This is required for Nsight for Eclipse
                export JAVA_HOME="${pkgs.jdk8.home}"
                # It needs libGL and libGLU (FIXME could be simlinked)
                export GLPATH=/usr/lib

                # Nvidia offload
                export __NV_PRIME_RENDER_OFFLOAD=1
                export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0
                export __GLX_VENDOR_LIBRARY_NAME=nvidia
                export __VK_LAYER_NV_optimus=NVIDIA_only
              '';
            };
        }
      );
}
