FROM python:3.11

WORKDIR /app
COPY ./statistics /app
COPY ./pyproject.toml /app

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

CMD ["/bin/bash"]
