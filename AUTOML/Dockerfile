FROM python:3.10-slim

# Evita prompts interativos do apt
ENV DEBIAN_FRONTEND=noninteractive

# 1) Dependências de sistema (Java pro H2O, compiladores pro AutoGluon etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    default-jre-headless \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# 2) JAVA_HOME (ajusta pro path dessa imagem)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# 3) Pasta de trabalho
WORKDIR /workspace

# 4) Requisitos Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 5) Copiar o projeto todo
COPY . .

CMD ["bash"]
