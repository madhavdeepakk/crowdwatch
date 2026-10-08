FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY crowdwatch ./crowdwatch
COPY configs ./configs
RUN pip install --no-cache-dir .

# Counts and alerts are kept here; mount a volume to keep them between runs.
VOLUME /app/data
EXPOSE 8000

# The simulated mall by default. For a camera:
#   docker run -p 8000:8000 -v $PWD/configs:/app/configs crowdwatch \
#     crowdwatch run configs/my_camera.yaml --host 0.0.0.0 --no-browser
CMD ["crowdwatch", "demo", "--host", "0.0.0.0", "--no-browser"]
