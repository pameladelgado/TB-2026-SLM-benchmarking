### Hardware platforms used for evaluation.

| Platform       | CPU                       | RAM     | GPU                       | VRAM               | OS                  | Backend            |
|----------------|---------------------------|---------|---------------------------|--------------------|---------------------|--------------------|
| Raspberry Pi 5 | Cortex-A76, 4 cores       | 7.9 GiB | none                      | —                  | Debian 13 (trixie)  | llama.cpp          |
| Laptop CPU     | Intel i5-1345U, 10c/12t   | 30 GiB  | integrated Intel          | —                  | OpenSUSE Tumbleweed | llama.cpp (Podman) |
| GPU server     | 2× Xeon 6520P, 96 threads | 251 GiB | 2× RTX PRO 6000 Blackwell | 97,887 MiB per GPU | Ubuntu 24.04.3 LTS  | vLLM 0.20.0        |
