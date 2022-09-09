FROM python:3.9.7-slim-buster

WORKDIR /src/

COPY requirements.txt requirements.txt

RUN pip3 install Cython

# RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


# RUN pip3 install hydra-core --upgrade

RUN pip3 install -r requirements.txt

COPY . .

# ENTRYPOINT ["python3", "main.py"]