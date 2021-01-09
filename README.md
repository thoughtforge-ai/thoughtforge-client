# thoughtforge-client
Client SDK for ThoughtForge AI

Installation instructions:
1) Clone repo: git clone git@github.com:thoughtforge-ai/thoughtforge-client.git thoughtforge_client
Note: Python has difficulty with root directories that contain dashes, so it's important to put into a folder named thoughtforge_client (or some other python-module-safe name)
2) cd thoughtforge_client
3) pip install -r requirements.txt
4) set THOUGHTFORGE_API_KEY environment variable to your API key
5) set server THOUGHTFORGE_HOST and THOUGHTFORGE_PORT environment variables as needed
(optionally, the client makes use of python-dotenv, and you can place your environment variables in a .env file)

API documentation can be found at https://thoughtforge-api.readthedocs.io/

To build and view documentation locally:
1) cd docs
2) make html
3) navigate to docs/build/html/index.html in your browser to see the compiled docs
