FROM ubuntu:latest 
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip cmake
WORKDIR /home/cristian/PycharmProjects/tEDRAM/tEDRAM2
ENV VIRTUAL_ENV=/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/dependencies
RUN python3.10 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY ./run.sh ./

COPY ./setup.py ./
COPY ./setup.cfg ./
COPY ./myproject.toml ./
COPY ./requeriments.txt ./

# RUN python3.10 -m pip install -e ./
RUN python3.10 -m pip install -r ./requeriments.txt

WORKDIR /app