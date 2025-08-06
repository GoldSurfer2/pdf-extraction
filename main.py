import os
from pdf2image import convert_from_path
import easyocr
import json
import numpy as np
from PIL import Image
import cv2
from google.cloud import documentai_v1 as documentai

PDF_PATH = 'pdf/1.발표자료.pdf'

import datetime

# 결과물 저장 폴더 및 파일명에 타임스탬프 추가
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_DIR = f'result_{timestamp}'
OUTPUT_DIR = os.path.join(RESULT_DIR, 'images')
OCR_RESULT_PATH = os.path.join(RESULT_DIR, f'ocr_result_{timestamp}.txt')
OCR_JSON_PATH = os.path.join(RESULT_DIR, f'ocr_result_{timestamp}.json')

# 선택할 페이지 번호(1부터 시작, 여러 개 가능)
SELECTED_PAGES = [3]#, 4, 6, 8, 10, 12, 25, 27, 29, 31]  # 3페이지: 다이어그램 | 4페이지: 간단한 표 | 6페이지: 타임라인 | 8페이지: 수치데이터표 | 10페이지: 지도 | 12페이지: 수식 | 25페이지: 그래픽 | 27페이지: ocr 되지 않는 텍스트 많음 | 29페이지: 그래프 많음 | 31페이지: 수식 | 

# 1. PDF → 이미지 변환 (선택한 페이지만)
os.makedirs(OUTPUT_DIR, exist_ok=True)
POPLER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'poppler-24.08.0', 'Library', 'bin'))
images = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPLER_PATH)
image_paths = []
selected_images = []
for idx, img in enumerate(images):
    page_num = idx + 1
    if page_num in SELECTED_PAGES:
        img_path = os.path.join(OUTPUT_DIR, f'page_{page_num}.png')
        # 파일 저장 생략
        image_paths.append(img_path)
        selected_images.append((page_num, img_path, img))  # img는 PIL.Image 객체

# 2. EasyOCR로 한글 OCR (선택한 페이지만)
reader = easyocr.Reader(['ko', 'en'], gpu=False)
ocr_results = []
def remove_meta_info(text):
    lines = text.split('\n')
    # 제목은 첫 줄, 그 다음부터 'K'가 단독으로 나오는 줄까지 제거
    new_lines = [lines[0]]
    k_found = False
    for line in lines[1:]:
        if not k_found:
            if line.strip() == 'K':
                k_found = True
                continue  # 'K' 줄까지 모두 제거, 해당 줄도 포함
        if k_found:
            new_lines.append(line)
    return '\n'.join(new_lines)

# OCR 부분에서 img_path 대신 img(PIL.Image)를 직접 전달
for page_num, img_path, img in selected_images:
    img_np = np.array(img)  # PIL.Image → numpy array 변환
    result = reader.readtext(img_np, detail=0, paragraph=True)
    page_text = '\n'.join(result)
    filtered_text = remove_meta_info(page_text)
    ocr_results.append({'page': page_num, 'image': img_path, 'text': filtered_text})

# 3. 결과 저장 (텍스트 + JSON)
with open(OCR_RESULT_PATH, 'w', encoding='utf-8') as f:
    for item in ocr_results:
        f.write(f'--- Page {item["page"]} ---\n')
        f.write(item['text'])
        f.write('\n\n')

# 선택한 페이지만 JSON으로 저장
with open(OCR_JSON_PATH, 'w', encoding='utf-8') as jf:
    json.dump(ocr_results, jf, ensure_ascii=False, indent=2)

print(f"OCR 결과가 {OCR_RESULT_PATH}와 {OCR_JSON_PATH}에 저장되었습니다.")

# OpenCV 기반 간단한 구조 분석 (Windows 호환)
def simple_layout_analysis(img_np):
    """OpenCV를 사용한 간단한 레이아웃 분석"""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 텍스트 블록 감지를 위한 형태학적 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elements = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # 최소 크기 필터링
            elements.append({
                "type": "text_block",
                "coordinates": [x, y, x+w, y+h],
                "text": None,
                "score": cv2.contourArea(contour)
            })
    
    return elements

layout_results = []
for page_num, img_path, img in selected_images:
    img_np = np.array(img)
    elements = simple_layout_analysis(img_np)
    
    ocr_item = next((item for item in ocr_results if item["page"] == page_num), None)
    layout_results.append({
        "page": page_num,
        "image": img_path,
        "layout_elements": elements,
        "ocr_text": ocr_item["text"] if ocr_item else ""
    })

LAYOUT_JSON_PATH = os.path.join(RESULT_DIR, f'layout_result_{timestamp}.json')
with open(LAYOUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(layout_results, f, ensure_ascii=False, indent=2)
print(f"[구조 분석 결과] Layout 구조 정보가 {LAYOUT_JSON_PATH}에 저장되었습니다.")


# ========== Google Document AI Layout Parser 방식 ==========
# PDF 원본에서 OCR+레이아웃 분석을 한 번에 처리
GOOGLE_RESULT_PATH = os.path.join(RESULT_DIR, f'google_layout_result_{timestamp}.json')
GOOGLE_OCR_PATH = os.path.join(RESULT_DIR, f'google_ocr_result_{timestamp}.txt')

project_id = "intense-dolphin-467311-r2"
location = "us"  # 또는 eur, asia
processor_id = "c6007bbdcd842b38"  # 콘솔에서 프로세서 생성 필요

# 원하는 페이지만 추출하려면 별도 PDF를 만들거나, 전체 PDF를 보내고 결과에서 원하는 페이지만 필터링
PDF_FOR_GOOGLE = PDF_PATH  # 전체 PDF 사용

client = documentai.DocumentUnderstandingServiceClient()
name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

with open(PDF_FOR_GOOGLE, "rb") as document:
    document_content = document.read()

raw_document = documentai.RawDocument(content=document_content, mime_type="application/pdf")
request = documentai.ProcessRequest(name=name, raw_document=raw_document)
result = client.process_document(request=request)

# 결과 파싱: 페이지별 텍스트 및 레이아웃 정보 추출
google_layout_results = []
google_ocr_results = []
doc = result.document
for page_idx, page in enumerate(doc.pages):
    page_num = page.page_number  # 1부터 시작
    if page_num not in SELECTED_PAGES:
        continue
    # OCR 텍스트 추출
    page_text = doc.text[page.layout.text_anchor.text_segments[0].start_index:page.layout.text_anchor.text_segments[0].end_index] if page.layout.text_anchor.text_segments else ""
    google_ocr_results.append({'page': page_num, 'text': page_text})
    # 레이아웃 요소 추출
    elements = []
    for block in page.blocks:
        coords = [
            int(block.layout.bounding_poly.vertices[0].x),
            int(block.layout.bounding_poly.vertices[0].y),
            int(block.layout.bounding_poly.vertices[2].x),
            int(block.layout.bounding_poly.vertices[2].y)
        ]
        block_text = doc.text[block.layout.text_anchor.text_segments[0].start_index:block.layout.text_anchor.text_segments[0].end_index] if block.layout.text_anchor.text_segments else ""
        elements.append({
            "type": block.layout.type_.name if hasattr(block.layout, 'type_') else "block",
            "coordinates": coords,
            "text": block_text,
            "score": block.layout.confidence
        })
    google_layout_results.append({
        "page": page_num,
        "layout_elements": elements,
        "ocr_text": page_text
    })

# 결과 저장
with open(GOOGLE_RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(google_layout_results, f, ensure_ascii=False, indent=2)
with open(GOOGLE_OCR_PATH, "w", encoding="utf-8") as f:
    for item in google_ocr_results:
        f.write(f'--- Page {item["page"]} ---\n')
        f.write(item['text'])
        f.write('\n\n')
print(f"[구글 DocumentAI 결과] Layout 구조 정보가 {GOOGLE_RESULT_PATH}에 저장되었습니다.")
print(f"[구글 DocumentAI 결과] OCR 텍스트가 {GOOGLE_OCR_PATH}에 저장되었습니다.")
