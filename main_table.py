from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from helper_functions import *
import os
import pandas as pd

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/logos")
async def logo(request: Request):
    return templates.TemplateResponse("logo.html", {"request": request})

@app.get("/table-qa-bot")
async def table(request: Request):
    return templates.TemplateResponse("table.html", {"request": request})

@app.post("/chatbot")
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    UPLOAD_DIR = 'static'
    save_path = os.path.join(UPLOAD_DIR, f"temp.pdf")
    with open(save_path, "wb") as f:
        f.write(pdf_file.file.read())
    global tables, i, n, flag, history
    tables = extract_all_tables(save_path)
    i = 0
    flag = True
    static_memory = []
    for table in tables:
        if type(table)==type(pd.DataFrame()) and len(table)!=0:
            static_memory.append(table)
    tables = static_memory
    n = len(tables)
    history = "<p> <b class='bold'>TABVISION'S BOT</b>> Number of tables Extracted: "+ str(n) + "</p> <p> Table No. "+ str(i+1)+ "</p>"+ tables[i].to_html()
    return templates.TemplateResponse("chatbot.html", {"request": request, "message": history})

@app.post("/chatbegins")
async def chat(request: Request, query: str = Form(...)):
    global tables, i, n, flag, history
    if n == 0:
        history += "<p> <b class='bold'>TABVISION'S BOT</b>> No tables detected 1"
        return templates.TemplateResponse("chatbot.html", {"request": request, "message": history})
    data = tables[i]
    history += '<br><p> <b class="bold">QUERY</b>: ' + query + '</p>'
    if query == "next":
        i = (i + 1)%n
        while(type(tables[i])!=type(pd.DataFrame())):
            i = (i+1)%n
        history += "<p> <b class='bold'>TABVISION'S BOT</b>> Table No. " + str(i+1) + "</p>" + tables[i].to_html()
        return templates.TemplateResponse("chatbot.html", {"request": request, "message": history})
    send_data = qa_bot_on_table(data, query)
    history += "<p> <b class='bold'>TABVISION'S BOT</b>> " + send_data + "</p>"
    return templates.TemplateResponse("chatbot.html", {"request": request, "message": history})

# To run the app:
# For logos (icevision): uvicorn main_logos:app --reload --port 9000
# For table-qa-bot (torch): uvicorn main_table:app --reload --port 8000