FROM idein/chainer:6.2.0
RUN pip install jupyter jupytext
RUN mkdir /root/.jupyter && echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
