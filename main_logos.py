from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import  FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from logo_removal import *
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/logos")
async def logo(request: Request):
    return templates.TemplateResponse("logo.html", {"request": request})

@app.post("/logos")
async def login(request: Request, pdf_file: UploadFile = File(...)):
    UPLOAD_DIR = 'static'
    save_path = os.path.join(UPLOAD_DIR, f"temp.pdf")
    with open(save_path, "wb") as f:
        f.write(pdf_file.file.read())
    remove_logos(save_path)
    return templates.TemplateResponse("logo.html", {"request": request})

@app.get("/table-qa-bot")
async def table(request: Request):
    return templates.TemplateResponse("table.html", {"request": request})

@app.post("/download-pdf")
def download_pdf():
    pdf_path = "static/removed_logo.pdf"  # Adjust the path to your PDF file
    return FileResponse(pdf_path, filename="removed_logo.pdf")

# To run the app:
# For logos (icevision): uvicorn main:app --reload --port 9000
# For table-qa-bot (torch): uvicorn main:app --reload --port 8000