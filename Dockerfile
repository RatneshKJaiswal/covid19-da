FROM apache/spark:3.5.3

USER root
RUN pip install --no-cache-dir pandas pyarrow
USER spark

