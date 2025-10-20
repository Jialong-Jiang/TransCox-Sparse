# TransCox-Sparse Docker 镜像
FROM rocker/r-ver:4.3.0

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建 Python 虚拟环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装 Python 依赖
RUN pip install --no-cache-dir numpy torch

# 复制项目文件
COPY . /app/

# 安装 R 依赖
RUN Rscript setup_environment.R

# 设置环境变量
ENV R_LIBS_USER=/usr/local/lib/R/site-library

# 暴露端口（如果需要 Shiny 应用）
EXPOSE 3838

# 默认命令
CMD ["R"]