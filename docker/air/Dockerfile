# syntax=docker/dockerfile:1
# Databricks AI Runtime 에서 `km run` 에 필요한 패키지를 미리 설치한 이미지.
# 실행마다 pip 설치를 하지 않도록 `km databricks` 가 environment.unity_catalog_image 로 사용 (databricks.local.yaml 의 image).
#
# 빌드·업로드 (README 의 "Databricks 실행 이미지" 참고):
#   docker build --platform linux/amd64 --provenance=false --sbom=false \
#     --secret id=uv,src=$HOME/.config/uv/uv.toml -t km-air:v4 docker/air
#   databricks air images push --source km-air:v4 --catalog <catalog> --schema <schema> --artifact km-air:v4
# pypi.org 가 막힌 네트워크에서는 uv 설정(사내 PyPI 프록시)을 secret 으로 넘김 → 이미지에는 남지 않음.
# secret 이 없으면 uv 기본값(pypi.org) 사용.
# --provenance/--sbom=false: buildx 기본 attestation 이 붙으면 OCI index 가 되어 Artifact Registry 가
# manifest 등록에서 500 을 냄 (레이어 업로드는 됨) → 단일 manifest 로 빌드.
FROM databricksruntime/air:dcs-base-aws-runtime

COPY requirements.txt /tmp/requirements.txt
# LimiX 는 torch==2.9.1 고정을 피하려고 의존성 없이 설치 (의존성은 requirements.txt 에 있음)
RUN --mount=type=secret,id=uv,target=/run/secrets/uv.toml \
    if [ -f /run/secrets/uv.toml ]; then export UV_CONFIG_FILE=/run/secrets/uv.toml; fi; \
    uv pip install --no-cache -r /tmp/requirements.txt && \
    uv pip install --no-cache --no-deps \
      "git+https://github.com/limix-ldm-ai/LimiX.git@516bf396333feb3198cf7aff8a6c10421f218e24"
# 베이스 이미지에 없는 시스템 패키지. 큰 패키지 레이어 캐시를 깨지 않도록 뒤에 둠
#   libgomp1: lightgbm 의 OpenMP 런타임
#   gcc, libc6-dev, python3.12-dev: triton 이 커널 런처(cuda_utils.c)를 컴파일 (kumo). Python.h 는 /usr/include/python3.12
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 gcc libc6-dev python3.12-dev \
    && rm -rf /var/lib/apt/lists/*
