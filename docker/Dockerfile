FROM python:3.10-alpine as base

ENV POETRY_VERSION=${POETRY_VERSION:-1.7.0}

# Install required apk packages
RUN apk add build-base curl
# ----------------------------------------------------------------------------------
FROM base as poetry-base

RUN pip3 install "poetry==$POETRY_VERSION" &&\
    poetry config virtualenvs.create false

# ----------------------------------------------------------------------------------
FROM poetry-base as app-base

COPY pyproject.toml poetry.lock README.md /grumpy_or_not/
COPY models/* /grumpy_or_not/models
COPY web_app/* /grumpy_or_not/web_app

WORKDIR /grumpy_or_not
RUN poetry install --only main --no-interaction --no-ansi

# ----------------------------------------------------------------------------------
FROM app-base as main

EXPOSE ${GRUMPY_PORT:-8080}
HEALTHCHECK --interval=2s --timeout=3s --retries=10\
	CMD curl --fail http://${GRUMPY_HOST:-0.0.0.0}:${GRUMPY_PORT:-8080} || exit 1

ENTRYPOINT [ "python" ]
CMD ["app.py"]
