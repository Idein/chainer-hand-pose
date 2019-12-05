# How to use this  Compose File

- verify you can use command `docker-compose`. If not, install docker-compose.
- Please refer https://docs.docker.com/compose/install/

# build

```
$ cd /path/to/folder/this/README.md
$ docker-compose build
```

# How to run

## easy method

```
$ docker-compose up
```

- or you can specify service by the following proceduce:

## run service `python`

```
$ docker-compose run --rm -w /work python python3
Python 3.5.2 (default, Nov 12 2018, 13:43:14)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> # do something what you want
>>> import os
>>> os.getcwd()
'/work'
```

## run service `jupyter`

- You can initialize jupyter notebook:

```
$ docker-compose run --rm -p 8888:8888 jupyter
[W 09:37:52.208 NotebookApp] Config option `default_jupytext_formats` not recognized by `LargeFileManager`.
[I 09:37:52.212 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 09:37:52.356 NotebookApp] All authentication is disabled.  Anyone who can connect to this server will be able to run code.
[I 09:37:52.387 NotebookApp] [Jupytext Server Extension] Deriving a JupytextContentsManager from LargeFileManager
[I 09:37:52.388 NotebookApp] Serving notebooks from local directory: /work
[I 09:37:52.389 NotebookApp] The Jupyter Notebook is running at:
[I 09:37:52.389 NotebookApp] http://xxxxxxxxx:8888/
[I 09:37:52.389 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 09:37:52.391 NotebookApp] No web browser found: could not locate runnable browser.
```
- Then open your web browser and go to `localhost:8888`

# clean up

```
$ docker-compose down
```