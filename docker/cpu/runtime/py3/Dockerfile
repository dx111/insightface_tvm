FROM dx9527/tvm:py3-20.07.17

COPY src /usr/local/app/src

COPY model /usr/local/app/model

WORKDIR /usr/local/app/src

RUN python -m compileall -b . \
    && find -name "*.py" | xargs rm -f \
    && find -name "__pycache__" | xargs rm -rf

CMD uvicorn main:app --host 0.0.0.0 --port 8018

