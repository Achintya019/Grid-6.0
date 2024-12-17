FROM python:3.12

ENV PYTHONUNBUFFERED 1

EXPORT PORT=8080

ENV APP_HOME /app 
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port 8080 --server.enableCORS false
