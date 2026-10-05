ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

ARG TF_REF
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
LABEL tensorfold.ref=${TF_REF}

RUN pip install --no-cache-dir "tensorfold[vision] @ git+${TF_REPO}@${TF_REF}" \
 && tensorfold --version

ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton

WORKDIR /workspace
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "tensorfold"]
