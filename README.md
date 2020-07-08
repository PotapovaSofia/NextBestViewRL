# NextBestViewRL

```bash
$ docker build -t nbv-experiments dockerfiles/
$ docker run --gpus=all -it -p PORT:PORT -v tensorflow_logs:/tmp/tensorflow_logs -v $PWD:/tf --name NAME --cpuset-cpus='0-9' --memory=32gb --memory-swap=32gb nbv-experiments
$ docker attach NAME
$ git clone --recursive https://github.com/mmolero/pypoisson.git
$ cd pypoisson
$ python setup.py build
$ python setup.py install
```
