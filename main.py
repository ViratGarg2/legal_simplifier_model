from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class SummarizationRequest(BaseModel):
    text: str
    min_len: Optional[int] = 50
    max_len: Optional[int] = 150

# Initialize model and tokenizer
MODEL_PATH = "your-username/legal-summarizer"  # Replace with your username
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
except Exception as e:
    # Fallback to local model if Hugging Face model is not available
    tokenizer = AutoTokenizer.from_pretrained("./legal-summarizer")
    model = AutoModelForSeq2SeqLM.from_pretrained("./legal-summarizer")

@app.post("/summarize/")
async def summarize_text(request: SummarizationRequest):
    try:
        # Tokenize and generate summary
        inputs = tokenizer(request.text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=request.max_len,
            min_length=request.min_len,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Legal Text Summarization API. Use POST /summarize/ to summarize text."}