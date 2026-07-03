FROM ghcr.io/ggml-org/llama.cpp:server

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    bash \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
COPY configs /app/configs
COPY runner /app/runner
COPY docker/entrypoint.sh /entrypoint.sh

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV BENCH_CONFIG="configs/llama_cpp_container.yaml"

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]