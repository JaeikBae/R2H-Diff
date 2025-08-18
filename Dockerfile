FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install flash-attn --no-build-isolation
RUN apt-get update --fix-missing
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 git vim wget curl

# Install dependencies
WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt
