FROM python:3.9-slim
COPY . /app
WORKDIR /app
EXPOSE 5000
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]