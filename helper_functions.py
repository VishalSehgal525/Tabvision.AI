from transformers import (DetrImageProcessor, TableTransformerForObjectDetection)
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from transformers import pipeline
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from itertools import count, tee
import torch
import matplotlib.pyplot as plt
import string
from tabula import read_pdf
import fitz
import os

# Loading all Models
table_detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
table_recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
config = Cfg.load_config_from_name('vgg_seq2seq')
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
detector = Predictor(config)
pipe = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")

# Helper functions for table extraction
def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def pytess(cell_pil_img):
    text, prob = detector.predict(cell_pil_img, return_prob=True)
    if prob < 0.5:
        return ""
    return text.strip()

def sharpen_image(pil_img):
    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img

def uniquify(seq, suffs=count(1)):
    not_unique = [k for k, v in Counter(seq).items() if v > 1]
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix
    return seq

def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
    result = cv2.GaussianBlur(thresh, (5, 5), 0)
    result = 255 - result
    return cv_to_PIL(result)

def td_postprocess(pil_img):
    # Removes gray background from tables
    img = PIL_to_cv(pil_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))  # (0, 0, 100), (255, 5, 255)
    nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))  # (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))  # (3,3)
    mask = mask & nzmask
    new_img = img.copy()
    new_img[np.where(mask)] = 255
    return cv_to_PIL(new_img)

def table_detector(image):
    # Table detection using DEtect-object TRansformer pre-trained on 1 million tables
    THRESHOLD_PROBA = 0.6
    feature_extractor = DetrImageProcessor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = table_detection_model(**encoding)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    return (probas[keep], bboxes_scaled)

def table_struct_recog(image):
    THRESHOLD_PROBA = 0.8
    feature_extractor = DetrImageProcessor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = table_recognition_model(**encoding)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    return (probas[keep], bboxes_scaled)

def plot_results_detection(model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
    # crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        xmin, ymin, xmax, ymax = xmin - delta_xmin, ymin - delta_ymin, xmax + delta_xmax, ymax + delta_ymax
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin - 20, ymin - 50, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

def crop_tables(pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
    # crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
    cropped_img_list = []
    for _, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        xmin, ymin, xmax, ymax = xmin - delta_xmin, ymin - delta_ymin, xmax + delta_xmax, ymax + delta_ymax
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
        cropped_img_list.append(cropped_img)
    return cropped_img_list

def add_padding(pil_img, top, right, bottom, left, color=(255, 255, 255)):
    # Image padding as part of TSR pre-processing to prevent missing table edges
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

colors = ["red", "blue", "green", "yellow", "orange", "violet"]
def generate_structure(model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
    # plt.figure(figsize=(32, 20))
    # plt.imshow(pil_img)
    # ax = plt.gca()
    rows = {}
    cols = {}
    idx = 0
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
        cl = p.argmax()
        class_text = model.config.id2label[cl.item()]
        text = f'{class_text}: {p[cl]:0.2f}'
        # or (class_text == 'table column')
        # if (class_text == 'table row') or (class_text == 'table projected row header') or (class_text == 'table column'):
        #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=colors[cl.item()], linewidth=2))
        #     ax.text(xmin - 10, ymin - 10, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))
        if class_text == 'table row':
            rows['table row.' + str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax, ymax + expand_rowcol_bbox_bottom)
        if class_text == 'table column':
            cols['table column.' + str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax, ymax + expand_rowcol_bbox_bottom)
        idx += 1
    return rows, cols

def sort_table_featuresv2(rows: dict, cols: dict):
    # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
    rows_ = {
        table_feature: (xmin, ymin, xmax, ymax)
        for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])
    }
    cols_ = {
        table_feature: (xmin, ymin, xmax, ymax)
        for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])
    }
    return rows_, cols_

def individual_table_featuresv2(pil_img, rows: dict, cols: dict):
    for k, v in rows.items():
        xmin, ymin, xmax, ymax = v
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
        rows[k] = xmin, ymin, xmax, ymax, cropped_img
    for k, v in cols.items():
        xmin, ymin, xmax, ymax = v
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
        cols[k] = xmin, ymin, xmax, ymax, cropped_img
    return rows, cols

def object_to_cellsv2(master_row: dict, cols: dict, padd_left):
    cells_img = {}
    row_idx = 0
    new_cols = {}
    new_master_row = {}
    new_cols = cols
    new_master_row = master_row
    for k_row, v_row in new_master_row.items():
        _, _, _, _, row_img = v_row
        xmax, ymax = row_img.size
        xa, ya, xb, yb = 0, 0, 0, ymax
        row_img_list = []
        for idx, kv in enumerate(new_cols.items()):
            k_col, v_col = kv
            xmin_col, _, xmax_col, _, col_img = v_col
            xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
            xa = xmin_col
            xb = xmax_col
            if idx == 0:
                xa = 0
            if idx == len(new_cols) - 1:
                xb = xmax
            xa, ya, xb, yb = xa, ya, xb, yb
            row_img_cropped = row_img.crop((xa, ya, xb, yb))
            row_img_list.append(row_img_cropped)
        cells_img[k_row + '.' + str(row_idx)] = row_img_list
        row_idx += 1
    return cells_img, len(new_cols), len(new_master_row) - 1

def clean_dataframe(df):
    for col in df.columns:
        df[col] = df[col].str.replace("'", '', regex=True)
        df[col] = df[col].str.replace('"', '', regex=True)
        df[col] = df[col].str.replace(']', '', regex=True)
        df[col] = df[col].str.replace('[', '', regex=True)
        df[col] = df[col].str.replace('{', '', regex=True)
        df[col] = df[col].str.replace('}', '', regex=True)
    return df

def convert_df(df):
    return df.to_csv().encode('utf-8')

def create_dataframe(cell_ocr_res: list, max_cols: int, max_rows: int):
    headers = cell_ocr_res[:max_cols]
    new_headers = uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
    counter = 0
    cells_list = cell_ocr_res[max_cols:]
    df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)
    cell_idx = 0
    for nrows in range(max_rows):
        for ncols in range(max_cols):
            df.iat[nrows, ncols] = str(cells_list[cell_idx])
            cell_idx += 1
    for x, col in zip(string.ascii_lowercase, new_headers):
        if f' {x!s}' == col:
            counter += 1
    df = clean_dataframe(df)
    return df

def start_process(img, padd_top, padd_left, padd_bottom, padd_right, delta_xmin, delta_ymin, delta_xmax,
                  delta_ymax, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
    image=Image.fromarray(img).convert("RGB")
    probas, bboxes_scaled = table_detector(image)
    if bboxes_scaled.nelement() == 0:
        return ''
    # plot_results_detection(table_detection_model, image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)
    cropped_img_list = crop_tables(image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)
    for unpadded_table in cropped_img_list:
        table = add_padding(unpadded_table, padd_top, padd_right, padd_bottom, padd_left)
        probas, bboxes_scaled = table_struct_recog(table)
        rows, cols = generate_structure(table_recognition_model, table, probas, bboxes_scaled, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom)
        rows, cols = sort_table_featuresv2(rows, cols)
        master_row, cols = individual_table_featuresv2(table, rows, cols)
        cells_img, max_cols, max_rows = object_to_cellsv2(master_row, cols, padd_left)
        sequential_cell_img_list = []
        for k, img_list in cells_img.items():
            for img in img_list:
                sequential_cell_img_list.append(
                    pytess(cell_pil_img=img))
        cell_ocr_res = sequential_cell_img_list
        df=create_dataframe(cell_ocr_res, max_cols, max_rows)
        return df

def get_tables_from_pdf(path):
    os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk"
    tables = read_pdf(path, stream=True, multiple_tables=True, pages='all', encoding='utf-8')
    return tables

def get_images_from_pdf(path):
    images = []
    pdf_file = fitz.open(path)
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index] 
        for _, img in enumerate(page.get_images(), start=1): 
            # get the XREF of the image 
            xref = img[0] 
            # extract the image bytes 
            base_image = pdf_file.extract_image(xref) 
            image_bytes = base_image["image"]
            nparr = np.frombuffer(image_bytes, np.uint8)
            imageFile = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(imageFile)
    return images

def image_to_table(img):
    return start_process(img, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0)

def extract_table_data_from_pdf(file_path):
    images = get_images_from_pdf(file_path)
    df = []
    for image in images:
        df.append(image_to_table(image))
    return df

def extract_all_tables(file_path):
    data1 = extract_table_data_from_pdf(file_path)
    data2 = get_tables_from_pdf(file_path)
    for d in data1:
        data2.append(d)
    return data2

# Helper Functions for Table QA bot
def process_data_for_qa_bot(d):
    for k, v in d.items():
        if isinstance(v, dict):
            process_data_for_qa_bot(v)
        else:
            if type(v) == int:
                v = str(v)
            d.update({k: v})
    return d

def qa_bot_on_table(table, queries):
    data = pipe(process_data_for_qa_bot(table.to_dict()), queries)
    if type(data) == type([]):
        answers = []
        for l in data:
            answers.append(l['answer'])
        return answers
    else:
        return data['answer']

# Chatbot function
def chatbot(path_to_pdf):
    tables = extract_all_tables(path_to_pdf)
    print("Welcome to Tabvision.AI. Ask what you want to.")
    while(True):
        flag1 = 0
        i = 1
        for table in tables:
            print("Table: " + str(i))
            print(table)
            i = i + 1
            flag2 = 0
            while(True):
                query = [input("Enter your query or Enter 'next' for next table or Enter 'exit' to finish: ")]
                if query[0]=='next':
                    break
                elif query[0]=='exit':
                    flag2 = 1
                    break
                else:
                    print(qa_bot_on_table(table, query))
            if flag2 == 1:
                flag1 = 1
                break
        if flag1 == 1:
            break