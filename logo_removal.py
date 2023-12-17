import fitz
import numpy as np
import cv2
from LOGOS.utils.logodetect import logodetection

logo_model = logodetection()

def img_replace(page, xref, filename=None, stream=None, pixmap=None):
    if bool(filename) + bool(stream) + bool(pixmap) != 1:
        raise ValueError("Exactly one of filename/stream/pixmap must be given")
    doc = page.parent
    new_xref = page.insert_image(
        page.rect, filename=filename, stream=stream, pixmap=pixmap
    )
    doc.xref_copy(new_xref, xref)
    last_contents_xref = page.get_contents()[-1]
    doc.update_stream(last_contents_xref, b" ")

def remove_logos(path):
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
            bboxes = logo_model.predict(imageFile)
            for box in bboxes:
                x, y, w, h = box
                imageFile[y:y+h, x:x+w, :] = 255
            buff_path = "./static/imgs/img.jpg"
            cv2.imwrite(buff_path, imageFile)
            img_replace(page, xref, filename=buff_path)
    pdf_file.save("./static/removed_logo.pdf")