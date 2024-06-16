FROM python:3.11.2-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install -r requirements.in
RUN pip-compile --generate-hashes requirements.in

CMD ["/bin/bash"]
