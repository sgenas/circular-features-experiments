FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sgenas/circular-features-experiments.git
WORKDIR /app/circular-features-experiments
RUN pip3 install -r requirements.txt

CMD ["bash"]