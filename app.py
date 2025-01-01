#In terminal ty command: 
# cd yolov8-web-app-youtube 
# export FLASK_APP=CheckEnvironment.py

from flask import Flask
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv() 
app = Flask(__name__)
CORS(app) 

# Get environment variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

@app.route('/')
def hello_world():
    return 'Hello world!'


# if __name__ == '__main__':
#     os.environ.setdefault('FLASK_ENV', 'development')
#     app.run(debug=False, port=5001, host='0.0.0.0')

if __name__ == '__main__':
    app.run()