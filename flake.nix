{
  description = "kube-autotuner development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python314
              pkgs.uv
              pkgs.ruff
              pkgs.ty
              pkgs.go-task
              pkgs.lefthook
              pkgs.kubectl
            ];

            shellHook = ''
              export UV_PYTHON_DOWNLOADS=never
              export UV_PYTHON=${pkgs.python314}/bin/python3.14
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$VIRTUAL_ENV/bin:$PATH"
              export XDG_DATA_DIRS="$PWD/.venv/completions:''${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
              export FPATH="$PWD/.venv/completions/zsh/site-functions:''${FPATH:-}"

              task --silent bootstrap
            '';
          };
        });
    };
}
