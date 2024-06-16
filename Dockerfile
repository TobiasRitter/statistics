FROM python:3.11.2-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install pip-tools
RUN pip-compile --generate-hashes pyproject.toml

CMD ["/bin/bash"]
