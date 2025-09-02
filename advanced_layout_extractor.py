#!/usr/bin/env python3
"""
고도화된 PDF 레이아웃 추출기
계층적 구조 보존 + 다중 모델 앙상블 + 공간 관계 분석
"""

import logging
import json
import numpy as np
import io
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# 기존 라이브러리들
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# 추가 레이아웃 분석 라이브러리들
import fitz  # PyMuPDF
from PIL import Image
import cv2
from statistics import median

# OCR 엔진(pytesseract) 선택적 사용
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElementType(Enum):
    """문서 요소 타입 정의"""
    TITLE = "title"
    PARAGRAPH = "paragraph" 
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    REFERENCE = "reference"

@dataclass
class LayoutElement:
    """레이아웃 요소 정보"""
    element_type: ElementType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    content: str
    confidence: float
    page_num: int
    hierarchy_level: int = 0
    parent_id: str = None
    children_ids: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []

class AdvancedLayoutExtractor:
    """고도화된 레이아웃 추출기 (Detectron2 미사용, PyMuPDF + OpenCV + 선택적 Tesseract)"""
    
    def __init__(self, use_tesseract_ocr: bool = True, tesseract_languages: str = "kor+eng"):
        """초기화"""
        # 1. Docling 설정 (기존 + 추가 옵션)
        self.pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
            images_scale=3.0,  # 더 고해상도
            generate_page_images=True,
            generate_table_images=True,
            generate_picture_images=True,
            # 추가 옵션들
            do_picture_extraction=True,
            do_table_extraction=True,
            table_structure_options={
                "do_cell_matching": True,
                "mode": "accurate"  # 정확도 우선
            }
        )
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        # 2. OCR 설정 (선택적)
        self.use_tesseract_ocr = bool(use_tesseract_ocr and PYTESSERACT_AVAILABLE)
        self.tesseract_languages = tesseract_languages
        
        logger.info(
            "고도화된 레이아웃 추출기 초기화 완료 (Detectron2 미사용, Tesseract OCR: %s)",
            "활성" if self.use_tesseract_ocr else "비활성"
        )

    def extract_hierarchical_layout(self, pdf_path: str) -> Dict[str, Any]:
        """계층적 레이아웃 추출"""
        logger.info(f"계층적 레이아웃 추출 시작: {pdf_path}")
        
        results = {
            "pages": [],
            "global_hierarchy": [],
            "spatial_relationships": [],
            "extraction_metadata": {}
        }
        
        # 1. Docling으로 기본 구조 추출
        docling_result = self._extract_with_docling(pdf_path)
        
        # 2. PyMuPDF + OpenCV 기반 시각 레이아웃 분석
        layout_result = self._extract_with_visual_analysis(pdf_path)
        
        # 3. 공간 관계 분석
        spatial_relations = self._analyze_spatial_relationships(layout_result)
        
        # 4. 계층 구조 구축
        hierarchy = self._build_hierarchy(layout_result, spatial_relations)
        
        # 5. 결과 통합
        results = self._merge_results(docling_result, layout_result, hierarchy, spatial_relations)
        
        logger.info(f"계층적 레이아웃 추출 완료")
        return results

    def _extract_with_docling(self, pdf_path: str) -> Dict[str, Any]:
        """Docling으로 기본 구조 추출 (개선된 버전)"""
        try:
            conv_result = self.doc_converter.convert(pdf_path)
            
            # 마크다운 + 구조화된 정보
            markdown_text = conv_result.document.export_to_markdown()
            
            # 상세한 요소 정보 추출
            elements = []
            for element, level in conv_result.document.iterate_items():
                element_info = {
                    "type": type(element).__name__,
                    "content": getattr(element, 'text', ''),
                    "page": getattr(element, 'page_number', 0),
                    "level": level,
                    "bbox": getattr(element, 'bbox', None),
                    "confidence": getattr(element, 'confidence', 1.0)
                }
                elements.append(element_info)
            
            return {
                "markdown": markdown_text,
                "elements": elements,
                "total_pages": len(conv_result.document.pages),
                "method": "docling"
            }
            
        except Exception as e:
            logger.error(f"Docling 추출 실패: {e}")
            return {"error": str(e), "method": "docling"}

    def _extract_with_visual_analysis(self, pdf_path: str) -> Dict[str, Any]:
        """PyMuPDF 텍스트/이미지 블록 + OpenCV 테이블 검출 + 선택적 Tesseract OCR"""
        try:
            doc = fitz.open(pdf_path)
            pages: List[Dict[str, Any]] = []
            
            num_pages = getattr(doc, "page_count", len(doc))
            for page_num in range(num_pages):
                page = doc[page_num]
                page_rect = page.rect
                page_width_pt = float(page_rect.width)
                page_height_pt = float(page_rect.height)
                
                # 고해상도 이미지 생성
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                image_np = np.array(image)
                img_w, img_h = image.size
                
                # 페이지 텍스트/이미지 블록(dict)
                page_dict = page.get_text("dict")
                span_sizes: List[float] = []
                for block in page_dict.get("blocks", []):
                    if block.get("type") == 0:  # text
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                size_val = span.get("size")
                                if size_val is not None:
                                    try:
                                        span_sizes.append(float(size_val))
                                    except Exception:
                                        pass
                median_size = median(span_sizes) if span_sizes else 0.0
                
                elements: List[LayoutElement] = []
                # 텍스트/이미지 블록을 요소로 변환
                for block in page_dict.get("blocks", []):
                    bbox = block.get("bbox", [0, 0, 0, 0])  # [x0, y0, x1, y1]
                    x0, y0, x1, y1 = [float(b) for b in bbox]
                    # 0-1 정규화 (페이지 단위)
                    norm_bbox = (
                        x0 / page_width_pt if page_width_pt else 0.0,
                        y0 / page_height_pt if page_height_pt else 0.0,
                        x1 / page_width_pt if page_width_pt else 0.0,
                        y1 / page_height_pt if page_height_pt else 0.0,
                    )
                    
                    if block.get("type") == 0:  # text
                        # 텍스트 수집
                        texts: List[str] = []
                        block_max_size = 0.0
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                t = span.get("text", "")
                                if t:
                                    texts.append(t)
                                size_val = span.get("size")
                                try:
                                    block_max_size = max(block_max_size, float(size_val))
                                except Exception:
                                    pass
                        content = " ".join(texts).strip()
                        # 간단한 제목 판단: 폰트 크기가 중앙값 대비 충분히 크면 TITLE
                        if median_size > 0 and block_max_size >= (median_size * 1.2):
                            etype = ElementType.TITLE
                            conf = 0.85
                        else:
                            etype = ElementType.PARAGRAPH
                            conf = 0.9
                        
                        elements.append(LayoutElement(
                            element_type=etype,
                            bbox=norm_bbox,
                            content=content,
                            confidence=conf,
                            page_num=page_num,
                            hierarchy_level=0
                        ))
                    elif block.get("type") == 1:  # image
                        extracted_text = ""
                        if self.use_tesseract_ocr:
                            # 페이지 좌표 → 픽셀 좌표
                            scale_x = img_w / page_width_pt if page_width_pt else 1.0
                            scale_y = img_h / page_height_pt if page_height_pt else 1.0
                            x0_px = max(0, min(int(x0 * scale_x), img_w - 1))
                            y0_px = max(0, min(int(y0 * scale_y), img_h - 1))
                            x1_px = max(0, min(int(x1 * scale_x), img_w))
                            y1_px = max(0, min(int(y1 * scale_y), img_h))
                            if x1_px > x0_px and y1_px > y0_px:
                                crop = image_np[y0_px:y1_px, x0_px:x1_px]
                                try:
                                    extracted_text = pytesseract.image_to_string(
                                        crop, lang=self.tesseract_languages
                                    ).strip()
                                except Exception as ocr_e:
                                    logger.warning("Tesseract OCR 실패(page %d): %s", page_num, ocr_e)
                                    extracted_text = ""
                        
                        elements.append(LayoutElement(
                            element_type=ElementType.FIGURE,
                            bbox=norm_bbox,
                            content=extracted_text,
                            confidence=0.7,
                            page_num=page_num,
                            hierarchy_level=0
                        ))
                
                # OpenCV 기반 테이블 후보 검출 (픽셀 좌표 → 정규화)
                try:
                    table_boxes_px = self._detect_tables_with_opencv(image_np)
                    for (tx0, ty0, tx1, ty1) in table_boxes_px:
                        norm_table_bbox = (
                            tx0 / img_w if img_w else 0.0,
                            ty0 / img_h if img_h else 0.0,
                            tx1 / img_w if img_w else 0.0,
                            ty1 / img_h if img_h else 0.0,
                        )
                        elements.append(LayoutElement(
                            element_type=ElementType.TABLE,
                            bbox=norm_table_bbox,
                            content="",
                            confidence=0.65,
                            page_num=page_num,
                            hierarchy_level=0
                        ))
                except Exception as tbl_e:
                    logger.warning("테이블 검출 실패(page %d): %s", page_num, tbl_e)
                
                pages.append({
                    "page": page_num,
                    "elements": elements,
                    "image_size": (img_w, img_h)
                })
            
            doc.close()
            return {"pages": pages, "method": "visual_pymupdf"}
        except Exception as e:
            logger.error("시각 레이아웃 분석 실패: %s", e)
            return {"error": str(e), "method": "visual_pymupdf"}

    def _convert_layout_to_elements(self, layout, page_num: int, model_name: str) -> List[LayoutElement]:
        """LayoutParser 결과를 LayoutElement로 변환"""
        elements = []
        
        for i, block in enumerate(layout):
            # 요소 타입 매핑
            element_type = self._map_layout_type(block.type, model_name)
            
            # bbox 정규화 (0-1 범위)
            bbox = (
                block.block.x_1 / layout.width,
                block.block.y_1 / layout.width, 
                block.block.x_2 / layout.width,
                block.block.y_2 / layout.width
            )
            
            element = LayoutElement(
                element_type=element_type,
                bbox=bbox,
                content=getattr(block, 'text', ''),
                confidence=getattr(block, 'score', 0.0),
                page_num=page_num,
                hierarchy_level=0  # 나중에 계산
            )
            
            elements.append(element)
        
        return elements

    def _map_layout_type(self, layout_type: str, model_name: str) -> ElementType:
        """레이아웃 타입을 ElementType으로 매핑"""
        type_mapping = {
            # PubLayNet 매핑
            "Text": ElementType.PARAGRAPH,
            "Title": ElementType.TITLE,
            "List": ElementType.LIST,
            "Table": ElementType.TABLE,
            "Figure": ElementType.FIGURE,
            
            # TableBank 매핑  
            "Table": ElementType.TABLE,
            
            # Newspaper 매핑
            "Headline": ElementType.TITLE,
            "Advertisement": ElementType.FIGURE,
            "Photograph": ElementType.FIGURE
        }
        
        return type_mapping.get(layout_type, ElementType.PARAGRAPH)

    def _analyze_spatial_relationships(self, layout_result: Dict) -> List[Dict]:
        """공간 관계 분석 (위/아래, 좌/우, 포함 관계 등)"""
        relationships = []
        
        # 새로운 구조(pages 기반)와 구구조(layouts 기반) 모두 지원
        page_entries = []
        if "pages" in layout_result:
            page_entries = layout_result.get("pages", [])
        elif "layouts" in layout_result:
            # 구형 구조 호환 처리
            for page_data in layout_result.get("layouts", []):
                merged_elements = []
                for _, elems in page_data.get("layouts", {}).items():
                    merged_elements.extend(elems)
                page_entries.append({"page": page_data.get("page", 0), "elements": merged_elements})
        
        for page_data in page_entries:
            page_num = page_data.get("page", 0)
            all_elements: List[LayoutElement] = page_data.get("elements", [])
            
            # 페어별 관계 분석
            for i, elem1 in enumerate(all_elements):
                for j, elem2 in enumerate(all_elements):
                    if i >= j:
                        continue
                    
                    relation = self._compute_spatial_relation(elem1, elem2)
                    if relation:
                        relationships.append({
                            "page": page_num,
                            "element1_id": f"{page_num}_{i}",
                            "element2_id": f"{page_num}_{j}",
                            "relation": relation,
                            "confidence": 0.8  # 기본값
                        })
        
        return relationships

    def _compute_spatial_relation(self, elem1: LayoutElement, elem2: LayoutElement) -> str:
        """두 요소 간의 공간 관계 계산"""
        x1_1, y1_1, x2_1, y2_1 = elem1.bbox
        x1_2, y1_2, x2_2, y2_2 = elem2.bbox
        
        # 중심점 계산
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # 포함 관계 확인
        if (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2):
            return "contains"
        elif (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1):
            return "contained_by"
        
        # 수직 관계
        if abs(center1[0] - center2[0]) < 0.1:  # 거의 같은 x 좌표
            if center1[1] < center2[1]:
                return "above"
            else:
                return "below"
        
        # 수평 관계  
        if abs(center1[1] - center2[1]) < 0.1:  # 거의 같은 y 좌표
            if center1[0] < center2[0]:
                return "left_of"
            else:
                return "right_of"
        
        return None

    def _build_hierarchy(self, layout_result: Dict, spatial_relations: List[Dict]) -> Dict:
        """계층 구조 구축"""
        hierarchy = {
            "root_elements": [],
            "parent_child_relations": [],
            "reading_order": []
        }
        
        # TODO: 공간 관계를 바탕으로 계층 구조 구축
        # 1. 제목-본문 관계 식별
        # 2. 표-캡션 관계 식별  
        # 3. 섹션 구조 파악
        # 4. 읽기 순서 결정
        
        return hierarchy

    def _merge_results(self, docling_result: Dict, layout_result: Dict, 
                      hierarchy: Dict, spatial_relations: List[Dict]) -> Dict:
        """모든 결과 통합"""
        merged = {
            "extraction_methods": ["docling", "visual_pymupdf"],
            "docling_extraction": docling_result,
            "layout_analysis": layout_result,
            "spatial_relationships": spatial_relations,
            "hierarchical_structure": hierarchy,
            "confidence_scores": self._calculate_confidence_scores(layout_result),
            "layout_quality_metrics": self._calculate_quality_metrics(layout_result)
        }
        
        return merged

    def _calculate_confidence_scores(self, layout_result: Dict) -> Dict:
        """신뢰도 점수 계산"""
        scores = {
            "overall_confidence": 0.0,
            "per_page_confidence": [],
            "per_element_type_confidence": {}
        }
        
        # TODO: 구현
        return scores

    def _calculate_quality_metrics(self, layout_result: Dict) -> Dict:
        """레이아웃 품질 지표 계산"""
        metrics = {
            "element_detection_count": 0,
            "average_confidence": 0.0,
            "coverage_ratio": 0.0,  # 페이지 영역 대비 감지된 영역 비율
            "overlap_ratio": 0.0    # 요소 간 겹침 비율
        }
        
        # TODO: 구현
        return metrics

    def _detect_tables_with_opencv(self, image_np: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """OpenCV로 테이블 후보 영역 감지 (픽셀 좌표 반환)
        간단한 선 구조 기반 검출: 수평/수직 선 추출 후 합성 → 외곽선 탐지
        """
        if image_np is None or image_np.size == 0:
            return []
        try:
            rgb = image_np
            if len(rgb.shape) == 3:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = rgb
            # 이진화
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 커널 크기 설정
            h, w = th.shape[:2]
            horiz_kernel_len = max(10, w // 40)
            vert_kernel_len = max(10, h // 40)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
            # 선 추출
            detect_horizontal = cv2.erode(th, horizontal_kernel, iterations=1)
            detect_horizontal = cv2.dilate(detect_horizontal, horizontal_kernel, iterations=1)
            detect_vertical = cv2.erode(th, vertical_kernel, iterations=1)
            detect_vertical = cv2.dilate(detect_vertical, vertical_kernel, iterations=1)
            table_mask = cv2.bitwise_or(detect_horizontal, detect_vertical)
            # 후처리
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes: List[Tuple[int, int, int, int]] = []
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                if area < 5000:
                    continue
                # 너무 가늘거나 긴 라인 박스 제외
                if min(bw, bh) < 20:
                    continue
                boxes.append((x, y, x + bw, y + bh))
            return boxes
        except Exception:
            return []


# 사용 예시
if __name__ == "__main__":
    extractor = AdvancedLayoutExtractor()
    
    # PDF 처리
    pdf_path = "pdf/음료흘림방지기술.pdf"
    result = extractor.extract_hierarchical_layout(pdf_path)
    
    # 결과 저장
    with open("advanced_layout_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("고도화된 레이아웃 추출 완료!")
