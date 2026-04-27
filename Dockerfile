FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install uv && \
    uv pip install --system -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]