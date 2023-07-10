FROM python:3.10



#COPY /app .
COPY . .
COPY requirements.txt .
COPY requirements-dev.txt .

RUN chmod 1777 /tmp
RUN apt-get update -y
RUN apt-get update
RUN apt-get install cmake -y && apt-get install vim -y

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt


#CMD ["jupyter", "notebook", "--port=8890", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
CMD ["sleep", "infinity"]
