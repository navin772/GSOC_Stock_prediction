FROM python:3.10-slim
COPY /fin_dashboard /app
WORKDIR /app
EXPOSE 8501
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
