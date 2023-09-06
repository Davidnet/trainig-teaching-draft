FROM tensorflow/tensorflow:2.13.0-gpu-jupyter
WORKDIR /usr/src/app
COPY --chmod=a+rwx . /usr/src/app/
RUN python -m pip install -e .