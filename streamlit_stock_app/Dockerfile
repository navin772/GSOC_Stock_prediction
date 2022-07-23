FROM python:3.9-slim
COPY . /app
WORKDIR /app
EXPOSE 8501
RUN pip3 install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]