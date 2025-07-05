FROM python:3.10
RUN apt-get update -y

WORKDIR /app

COPY . /app

RUN pip install -r /app/requirements.txt

RUN echo $PYTHONPATH


# Create a startup script
RUN echo '#!/bin/bash\n\
python tele_producer.py &\n\
python tele_consumer.py &\n\
wait' > /app/start.sh

# Make the script executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]


