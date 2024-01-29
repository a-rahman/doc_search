# Doc Search

## Installation
1. Install Python (tested with python 3.10)

2. If using the chatbot, GPU support is needed. Install `pytorch` and `bitsandbytes` with GPU support. The following is the calls for windows:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes==0.39.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```
If not using the chatbot, or no gpu is available, skip this step.

3. Install the other requirements with:
```
pip install -r requirements.txt
```

## Usage

For the document search call:
```
python document_semantic_search.py
```
The local URL to the app will be on the console, and by default should be http://127.0.0.1:7860

![Alt text](imgs/doc_search.PNG?raw=true "Document Search")

- Multiple files can be uploaded at a time. On CPU, it can take some time to embed and index.
- Sources will list the files already in the vector database.

For the the chatbot call:
```
python chatbot.py
```
Default assumptions are that a local machine with a suitable GPU will be running the chatbot application so that other machines on the local network can do retrieval augmented generation. On the hosting machine, the app can be accessed at http://localhost:7860, otherwise at http://<ip_of_host_machine>:7860 from the local network.
![Alt text](imgs/chatbot.PNG?raw=true "Chatbot")

By default it uses mistral-7B as the language model and was tested serving with a GTX 3090.
