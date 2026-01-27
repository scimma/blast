FROM python:3.11.13 AS deps

RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    -o /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc

RUN pip install --upgrade pip
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir

FROM python:3.11.13
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
    firefox-esr \
    && rm -rf /var/lib/apt/list/*

# Mozilla geckodriver is required for thumbnail generation
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.36.0-linux64.tar.gz && \
    chmod +x geckodriver && \
    mv geckodriver /usr/local/bin/

COPY ./debug/debug_ipython.py /root/.ipython/profile_default/startup/
COPY --from=deps /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=deps /usr/local/bin/ /usr/local/bin/

# Install Prost from source
# TODO: Remove this when astro-prost package is updated
RUN git clone https://github.com/manning-ncsa/Prost.git /tmp/prost && cd /tmp/prost && \
    git checkout 4bdbdc7 && \
    pip uninstall --yes astro-prost && pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/prost

RUN mkdir /app
COPY . /app
WORKDIR /app
