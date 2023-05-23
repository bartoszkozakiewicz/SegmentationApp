FROM python:3.9.12

WORKDIR /usr/src/app

COPY main.py .
COPY model_addons.py .
COPY app.py .

COPY templates .
COPY static .
COPY model .
COPY __pycache__ .

COPY requirements.txt .

EXPOSE 5000

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]