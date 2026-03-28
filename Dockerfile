FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD [ "gunicorn", "app:app", "--bind", "0.0.0:10000" ]
