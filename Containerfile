FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /data

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["fastapi","dev", "--host","0.0.0.0", "webserver.py"]
