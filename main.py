from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load the model and tokenizer from the local path
local_path = "./legal-summarizer"
loaded_tokenizer = AutoTokenizer.from_pretrained(local_path)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(local_path)

app = FastAPI(title="Legal Text Summarizer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class SummarizationRequest(BaseModel):
    text: str
    min_len: int = 50  # default value
    max_len: int = 150  # default value

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Legal Text Summarizer API</title>
        </head>
        <body>
            <h1>Welcome to Legal Text Summarizer API</h1>
            <p>Use POST /summarize/ endpoint to summarize legal text</p>
            <p>Example request:</p>
            <pre>
            POST /summarize/
            {
                "text": "Your legal text here"
            }
            </pre>
        </body>
    </html>
    """

@app.get("/summarize/")
async def summarize_get():
    return JSONResponse(
        status_code=405,
        content={
            "detail": "Method not allowed. Please use POST request with JSON body containing 'text' field"
        }
    )

@app.post("/summarize/")
def summarize(request: SummarizationRequest):
    try:
        # Prepare the text for the model
        inputs = loaded_tokenizer.encode("summarize: " + request.text, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summary
        summary_ids = loaded_model.generate(
            inputs, 
            max_length=request.max_len, 
            min_length=request.min_len, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )

        # Decode the summary
        summary = loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))