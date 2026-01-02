# Overview
This simple integration demonstrates how one can build simple RAG document search. 
# Running the Document Query API
## Update the dependencies
pip install -r requirements.txt
## Update API Key
Update the OPENAI_API_KEY in .env file under root folder of this code 
## Make sure you have "docs" folder under the root project
## Run in local
uvicorn app.main:app --env-file .env --reload
