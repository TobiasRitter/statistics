FROM python:3.11.2-alpine

WORKDIR /app

COPY . .

RUN pip install -r requirements.in
RUN pip freeze > requirements.txt

CMD ["/bin/bash"]
