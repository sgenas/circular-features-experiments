FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

    CMD ["bash", "-c", "git clone your-github-repo-url && cd repo-name && pip3 install -r requirements.txt && bash"]
