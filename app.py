import os
from flask import Flask, render_template, request
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn
import pytesseract
import re
import easyocr
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

yolo_model = YOLO('best.pt')
reader = easyocr.Reader(['en', 'fr'], gpu=False)

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

digit_model = DigitCNN()
digit_model.load_state_dict(torch.load("cnn_digit_segment.pt", map_location='cpu'))
digit_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def corriger_index_manuellement(texte):
 
    if texte.startswith("4") and len(texte) <= 5:
        texte = "1" + texte[1:]
    
    return texte


def preprocess_digit(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return transform(thresh).unsqueeze(0)

def extract_info(img_path):
    img = cv2.imread(img_path)
    results = yolo_model(img)[0]

    detected_text = ""
    compteur_pred = ""

    for box in results.boxes.data:
        cls = int(box[5].item())
        label = results.names[cls]
        x1, y1, x2, y2 = map(int, box[:4])
        roi = img[y1:y2, x1:x2]

        if label == 'index':
            height = y2 - y1
            roi_top = roi[0:int(height * 0.9), :]

            gray = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            result = reader.readtext(thresh, allowlist='0123456789')
            roi_boxes_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            for (bbox, text, prob) in result:
                x_min = int(min(point[0] for point in bbox))
                y_min = int(min(point[1] for point in bbox))
                x_max = int(max(point[0] for point in bbox))
                y_max = int(max(point[1] for point in bbox))

                clean_text = re.sub(r'[^0-9]', '', text)
                if clean_text:
                    #clean_text = corriger_erreurs(clean_text)
                    clean_text = corriger_index_manuellement(clean_text)
                    detected_text += clean_text
                    cv2.rectangle(roi_boxes_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(roi_boxes_display, clean_text, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "detected_index.jpg")
            cv2.imwrite(output_path, roi_boxes_display)

        elif label == 'compteur':
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.bilateralFilter(roi_gray, 11, 17, 17)
            text = pytesseract.image_to_string(roi_gray, config='--psm 6')
            text = text.replace('O', '0').replace('I', '1').replace('|', '1')
            numbers = re.findall(r'\d{6,12}', text)
            if numbers:
                compteur_pred = max(numbers, key=len)

    return detected_text, compteur_pred

def generer_facture_pdf(index, compteur, montant, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 22)
    c.drawString(2 * cm, height - 3 * cm, "FACTURE D'ÉLECTRICITÉ")

    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, height - 4.5 * cm, f"Date : {datetime.now().strftime('%d/%m/%Y')}")
    c.drawString(2 * cm, height - 5.2 * cm, f"Numéro de compteur : {compteur}")
    c.drawString(2 * cm, height - 5.9 * cm, f"Index relevé : {index} kWh")

    
    data = [
        ['Description', 'Quantité (kWh)', 'Prix Unitaire', 'Montant Total'],
        ['Consommation estimée', index, '79 FCFA', montant]
    ]

    table = Table(data, colWidths=[7 * cm, 3 * cm, 3.5 * cm, 4 * cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9E1F2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#000000')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.8, colors.grey)
    ]))

    table.wrapOn(c, width, height)
    table.drawOn(c, 2 * cm, height - 10 * cm)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2 * cm, height - 13 * cm, " Ceci est une estimation automatique basée sur l’image envoyée.")
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, height - 14 * cm, "Merci d'utiliser notre service intelligent de lecture de compteur.")

    c.showPage()
    c.save()

def calculer_facture(index_detecte):
    try:
        index = int(index_detecte)/1000
    except ValueError:
        return "Index invalide", 0.0

    prix_unitaire = 79  
    montant = index * prix_unitaire
    return f"{montant:,.0f} FCFA".replace(',', ' '), montant


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

"""  
@app.route('/result', methods=['POST'])
def result():
    if 'image' not in request.files:
        return "Aucune image fournie"

    file = request.files['image']
    if file.filename == '':
        return "Nom de fichier vide"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    index, compteur = extract_info(file_path)

    return render_template('result.html',
                           index=index if index else "Non détecté",
                           compteur=compteur if compteur else "Non détecté",
                           image_path=file_path,
                           index_image="static/uploads/detected_index.jpg")
"""
@app.route('/result', methods=['POST'])
def result():
    if 'image' not in request.files:
        return "Aucune image fournie"

    file = request.files['image']
    if file.filename == '':
        return "Nom de fichier vide"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    index, compteur = extract_info(file_path)
    montant_str, montant_val = calculer_facture(index)

    
    pdf_filename = "facture.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    generer_facture_pdf(int(index)/1000, compteur, montant_str, pdf_path)

    return render_template('result.html',
                           index=index if index else "Non détecté",
                           compteur=compteur if compteur else "Non détecté",
                           montant=montant_str,
                           image_path=file_path,
                           index_image="static/uploads/detected_index.jpg",
                           facture_path=pdf_path)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
