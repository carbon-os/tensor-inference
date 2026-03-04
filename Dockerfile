# ==============================================================================
# STAGE 1: Builder (Compiles the C++/CUDA codebase and fetches the model)
# ==============================================================================
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

# Prevent tzdata and other interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for CMake, vcpkg, and your libraries
RUN apt-get update && apt-get install -y \
    git curl zip unzip tar pkg-config \
    build-essential cmake ninja-build \
    libssl-dev libcurl4-openssl-dev
