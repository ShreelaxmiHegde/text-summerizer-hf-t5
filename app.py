from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# initialize fastapi app
app = FastAPI(title="Text Summerizer", description="Text summerization app using T5", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# load model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./model/saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./model/saved_summary_model")

# define device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory=".")

# input schema for dialogue
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", "", text) # extra lines
    text = re.sub(r"\r+", "", text) # extra spaces
    text = re.sub(r"<.*?>", "", text) # html tags
    text = text.strip().lower() # trailing spaces
    return text

def summarize_dialogue(dialogue: str):
    print("summarizing dialogue...")
    dialogue = clean_data(dialogue) # clean input

    # tokenize input
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # generate summary (generates output in tokens form)
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )

    # decode output token ids to text
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# API endpoints
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})