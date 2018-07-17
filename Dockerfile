FROM python:3.6-onbuild

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--allow-root"]
