FROM registry.gitlab.com/ncsa-blast/kubernetes/blast/deps:20240912 AS deps

FROM python:3.11
ENV PYTHONUNBUFFERED=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    gfortran \
    libhealpix-cxx-dev \
    libhdf5-serial-dev \
    netcdf-bin \
    libnetcdf-dev \
    jq \
    xxd \
    && rm -rf /var/lib/apt/list/*

COPY ./debug/debug_ipython.py /root/.ipython/profile_default/startup/

COPY --from=deps /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=deps /usr/local/bin/ /usr/local/bin/

# [2024/09/12] This should be removed when Django fixes the bug as discussed in 
#              https://stackoverflow.com/a/78979171
RUN pip install importlib_metadata==8.4.0

RUN mkdir /app
COPY . /app
WORKDIR /app
