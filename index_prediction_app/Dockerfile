FROM python:3.10
COPY . /app
WORKDIR /app
EXPOSE 5000 30000-50000
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]