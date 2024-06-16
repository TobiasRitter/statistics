FROM python:3.11.2-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install pip-tools
RUN pip-compile --generate-hashes requirements.in

CMD ["/bin/bash"]
