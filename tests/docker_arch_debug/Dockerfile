FROM archlinux:latest

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
        curl \
        git \
        sudo \
        which \
        base-devel && \
    pacman -Scc --noconfirm

# Non-root user — conda/miniforge refuse to install as root
RUN useradd -m -s /bin/bash wbt && \
    echo "wbt ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER wbt
WORKDIR /workspace
