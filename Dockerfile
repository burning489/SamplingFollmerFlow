FROM python:3.11.6-slim-bullseye

WORKDIR /usr/src/fflow
RUN pip3 install --no-cache-dir numpy==1.23.4 matplotlib==3.7.1 scikit-learn==1.3.2 notebook==7.0.6 scipy==1.9.3 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
COPY . .

CMD jupyter-notebook --port 8888 --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' demo.ipynb