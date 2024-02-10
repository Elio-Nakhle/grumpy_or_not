FROM ubuntu:22.04 as base

ENV POETRY_VERSION=${POETRY_VERSION:-1.7.0}

RUN apt update &&\
    apt install python3 python3-pip wget curl -y

# ----------------------------------------------------------------------------------
FROM base as poetry-base

RUN pip3 install "poetry==$POETRY_VERSION" &&\
    poetry config virtualenvs.create false

# ----------------------------------------------------------------------------------
FROM poetry-base as app-base

COPY pyproject.toml poetry.lock README.md /grumpy_or_not/
COPY web_app/* /grumpy_or_not/web_app/
COPY models/* /grumpy_or_not/models/

WORKDIR /grumpy_or_not
RUN poetry install --only main --no-interaction --no-ansi

# ----------------------------------------------------------------------------------
FROM app-base as main

EXPOSE ${GRUMPY_PORT:-8080}
HEALTHCHECK --interval=2s --timeout=3s --retries=10\
	CMD curl --fail http://${GRUMPY_HOST:-0.0.0.0}:${GRUMPY_PORT:-8080} || exit 1

CMD gunicorn web_app.app:app -c web_app/gunicorn_prod.py
