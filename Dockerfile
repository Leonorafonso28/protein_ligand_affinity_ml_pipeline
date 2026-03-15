FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
    git wget build-essential \
    libglib2.0-0 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY . .

ENV PYTHONPATH=/app/src
ENV PATH /opt/conda/envs/protein_ligand_affinity/bin:$PATH

RUN mkdir -p data/raw data/interim data/features data/processed data/models

CMD ["python", "pipelines/pipeline_run_all.py"]