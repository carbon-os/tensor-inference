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

# Set up vcpkg permanently
ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git $VCPKG_ROOT && \
    $VCPKG_ROOT/bootstrap-vcpkg.sh

# Set the working directory to match your repo structure
WORKDIR /app

# Copy the entire libtransformers repository into the container
COPY . .

# Configure and compile the project (Using Ninja for faster parallel builds)
# We set CMAKE_BUILD_TYPE=Release to ensure -O3 and other optimizations are applied
RUN cmake -B build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DTRANSFORMERS_BUILD_TOOLS=ON \
    -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build

# Pre-fetch the model so it is baked into the image. 
# This prevents the container from re-downloading the model on every Cloud Run cold start.
RUN mkdir -p /root/.cache/models && \
    ./build/model-cli fetch hf://Qwen/Qwen2.5-Coder-1.5B-Instruct


# ==============================================================================
# STAGE 2: Runtime (Lightweight image containing only what is needed to run)
# ==============================================================================
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Cloud Run injects the PORT environment variable (default 8080)
ENV PORT=8080

# Ensure the GPU is visible and capabilities are exposed to the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install only the runtime shared libraries needed by your compiled binaries
RUN apt-get update && apt-get install -y \
    libssl3 libcurl4 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binaries from the builder stage
COPY --from=builder /app/build /app/build

# Copy the cached model from the builder stage
COPY --from=builder /root/.cache/models /root/.cache/models

# Document that this container listens on port 8080
EXPOSE 8080

# Start your hypothetical inference service and point it to the cached model
CMD ["./build/inference-service", "/root/.cache/models/Qwen/Qwen2.5-Coder-1.5B-Instruct"]