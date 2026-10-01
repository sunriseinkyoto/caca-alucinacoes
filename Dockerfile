# Ambiente de execução da solução Caça-Alucinações.
#
#   docker build -t caca-alucinacoes .
#   docker run --rm -v <pasta_dados>:/dados:ro -v <pasta_saida>:/saida \
#       caca-alucinacoes /dados/<base>.db /dados/txt /saida
#
# A imagem roda em CPU: a inferência do rotulador é leve, e o mesmo resultado
# é obtido em qualquer hardware. Para usar GPU, construa com
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
# e execute com --gpus all.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0

WORKDIR /app

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install --index-url "$TORCH_INDEX" torch==2.8.0 \
 && pip install -r requirements-docker.txt

# Pesos do rotulador (Release v1.0, SHA-256 conferido), baixados na construção
# da imagem: a execução não acessa a rede.
COPY baixar_modelo.py .
RUN python baixar_modelo.py

COPY . .

ENTRYPOINT ["bash", "/app/run.sh"]
