FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

COPY requirements.txt .

# Server deps
RUN pip install --no-cache-dir -r requirements.txt

# Upstream model deps (mirror diffusion-experiments/scripts/00_remote_setup.sh,
# minus the editable Open-dLLM install — we don't need that for inference)
RUN pip install --no-cache-dir \
      "transformers==4.54.1" accelerate "datasets>=3.0,<4.0" peft hf-transfer \
      "triton>=3.1.0" \
      codetiming hydra-core pandas "pyarrow>=15.0.0" pylatexenc \
      wandb "liger-kernel==0.5.8" \
      "diffusers>=0.30.0,<=0.31.0" tiktoken blobfile

COPY app.py inference.py postprocess.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
