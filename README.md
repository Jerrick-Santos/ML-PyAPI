# ML-PyAPI

# How to Run ML-PyAPI:

1. Ensure you have Python installed on your device. If not you can refer to https://www.python.org/downloads/ 
	- If you do not have python: Download Python version 3.10.2 or HIGHER
	- Check Python version using pip --version

2. Go to target directory by using cd 

3. Install dependencies using:

	pip install -r requirements.txt

	This may take about 3-5 minutes depending on your machine so please do wait until intallation is complete

4. Run API using
	
	python main.py 
	
	it should indicate this: "Running on http://127.0.0.1:5000" 
	Which indicates the URL to access the API

5. Ensure that the .env.local file matches the URL in the ML_API_URL variable to the given API URL
	
	ML_API_URL = http://127.0.0.1:5000

