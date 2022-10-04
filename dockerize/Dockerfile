FROM python:3.9-slim

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
#  && rm -rf /root/.cache/pip

COPY . .

EXPOSE 8080 

ENTRYPOINT ["python3", "demo_scripted.py"]
