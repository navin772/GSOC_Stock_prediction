FROM python:3.10
COPY . /app
WORKDIR /app
EXPOSE 8501
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]