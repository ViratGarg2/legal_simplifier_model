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
MODEL_PATH = os.getenv("MODEL_PATH", "./legal-summarizer")
try:
    print(f"Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

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
        print(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Legal Text Summarization API. Use POST /summarize/ to summarize text."}