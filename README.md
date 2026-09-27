# spetotextos

This is a small program for testing OCR, image processing, RAG, speech to text and text to speech.

It can be used to experiment with converting spoken audio into text and converting text back into spoken audio.

## Setup
### install 
python -m pip install playwright  

## Variables
```bash
$env:CHABELA_BASE_URL = "http://127.0.0.1:8000" 
$env:TESSERACT_CMD = "C:\Programs\Tesseract-OCR\tesseract.exe"
$env:TESSERACT_LANG = "eng"
```

## How to run
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## run tests
```bash
python -m pytest -q  
```
