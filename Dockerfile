FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./main.py /code/main.py
COPY ./legal-summarizer /code/legal-summarizer

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]