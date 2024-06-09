FROM python:3.11

WORKDIR /app
COPY ./statistics /app/statistics
COPY ./pyproject.toml /app
COPY ./README.md /app

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

CMD ["/bin/bash"]
