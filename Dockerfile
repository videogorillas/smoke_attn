
RUN virtualenv  -p python3 venv
RUN ./venv/bin/pip3 install -r requirements.txt 
