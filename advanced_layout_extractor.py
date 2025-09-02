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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR 엔진(pytesseract) 선택적 사용
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# AI 기반 분류기 모듈
try:
    from ai_layout_classifier import AILayoutClassifier
    AI_CLASSIFIER_AVAILABLE = True
except ImportError:
    AI_CLASSIFIER_AVAILABLE = False
    logger.warning("AI classifier module not found. Using fallback methods.")

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
        
        # 3. AI 분류기 초기화
        if AI_CLASSIFIER_AVAILABLE:
            self.ai_classifier = AILayoutClassifier()
            logger.info("GPT-4V 기반 AI 분류기 초기화 완료")
        else:
            self.ai_classifier = None
            logger.warning("AI 분류기 사용 불가, 기본 방식 사용")
        
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

    def _extract_with_ai_classification(self, pdf_path: str) -> Dict[str, Any]:
        """AI 기반 레이아웃 추출 (OpenCV 후보 검출 + GPT-4V 분류)"""
        logger.info("AI 기반 레이아웃 추출 시작")
        
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                
                # 고해상도 이미지 생성
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # numpy 배열로 변환
                image = Image.open(io.BytesIO(img_data))
                image_np = np.array(image)
                
                # 1단계: OpenCV로 후보 영역 검출 (단순화)
                candidate_bboxes = self._detect_candidate_regions_simple(image_np)
                
                # 2단계: AI로 영역 분류
                if self.ai_classifier and candidate_bboxes:
                    classifications = self.ai_classifier.classify_regions(image_np, candidate_bboxes)
                else:
                    # AI 분류기 없으면 모든 후보를 figure로 처리
                    classifications = [
                        {"bbox": bbox, "classification": "figure", "confidence": 0.5, "method": "fallback"}
                        for bbox in candidate_bboxes
                    ]
                
                # 3단계: 분류 결과를 LayoutElement로 변환
                elements = []
                for result in classifications:
                    bbox = result["bbox"]
                    classification = result["classification"]
                    confidence = result["confidence"]
                    
                    # 분류에 따른 ElementType 매핑
                    if classification == "table":
                        element_type = ElementType.TABLE
                    elif classification == "chart":
                        element_type = ElementType.FIGURE  # 차트도 figure로 처리
                    elif classification == "text":
                        element_type = ElementType.PARAGRAPH
                    else:  # figure
                        element_type = ElementType.FIGURE
                    
                    element = LayoutElement(
                        element_type=element_type,
                        bbox=bbox,
                        content="",
                        confidence=confidence,
                        page_num=page_idx,
                        hierarchy_level=0
                    )
                    elements.append(element)
                
                pages.append({
                    "page": page_idx,
                    "elements": elements
                })
                
                logger.info(f"페이지 {page_idx+1}: {len(candidate_bboxes)}개 후보 → {len(elements)}개 요소")
            
            doc.close()
            
            return {
                "pages": pages,
                "method": "ai_gpt4v"
            }
            
        except Exception as e:
            logger.error(f"AI 레이아웃 추출 실패: {e}")
            return {"pages": [], "method": "error"}

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
        개선된 선 구조 기반 검출: 수평/수직 선 추출 + 격자 교차점 검증 + 면적/비율 필터
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
            
            # 커널 크기 설정 (더 엄격하게)
            h, w = th.shape[:2]
            horiz_kernel_len = max(15, w // 30)  # 더 긴 선만 감지
            vert_kernel_len = max(15, h // 30)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
            
            # 선 추출
            detect_horizontal = cv2.erode(th, horizontal_kernel, iterations=1)
            detect_horizontal = cv2.dilate(detect_horizontal, horizontal_kernel, iterations=1)
            detect_vertical = cv2.erode(th, vertical_kernel, iterations=1)
            detect_vertical = cv2.dilate(detect_vertical, vertical_kernel, iterations=1)
            
            # 교차점 검출로 격자 구조 확인
            intersection = cv2.bitwise_and(detect_horizontal, detect_vertical)
            
            table_mask = cv2.bitwise_or(detect_horizontal, detect_vertical)
            
            # 후처리
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes: List[Tuple[int, int, int, int]] = []
            
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                
                # 강화된 필터링
                if area < 8000:  # 최소 면적 증가
                    continue
                if min(bw, bh) < 30:  # 최소 변 길이 증가
                    continue
                
                # 종횡비 제한 (너무 길거나 높은 박스 제외)
                aspect_ratio = max(bw, bh) / min(bw, bh)
                if aspect_ratio > 8:  # 8:1 비율 초과 제외
                    continue
                
                # 교차점 밀도 검사 (격자 구조 확인)
                roi_intersection = intersection[y:y+bh, x:x+bw]
                intersection_count = cv2.countNonZero(roi_intersection)
                intersection_density = intersection_count / area if area > 0 else 0
                
                if intersection_density < 0.0001:  # 교차점이 거의 없으면 제외
                    continue
                
                # 3D 다이어그램 vs 표 구분 (추가 검증)
                if not self._is_likely_table(x, y, bw, bh, detect_horizontal, detect_vertical, intersection):
                    continue
                
                # 새로운 증거 기반 분류기 사용 (우선)
                if self.table_figure_classifier:
                    region_bbox = [x/w, y/h, (x+bw)/w, (y+bh)/h]  # 정규화된 좌표
                    classification, score, features = self.table_figure_classifier.classify_region(
                        region_bbox, image_np, None, None
                    )
                    logger.info(f"증거 기반 분류기: {classification} (score: {score:.3f}, features: {features})")
                    if classification != "table":
                        logger.info(f"영역 제외됨: {classification} (score: {score:.3f})")
                        continue
                    else:
                        logger.info(f"표로 분류됨: score={score:.3f}")
                else:
                    # 텍스트 패턴 기반 표 검증 (폴백)
                    if not self._validate_table_content(image_np, x, y, bw, bh):
                        continue
                
                # 큰 영역은 여러 표로 분할 시도
                if bw > w * 0.6 and bh > h * 0.3:  # 페이지의 60% 너비, 30% 높이 초과
                    split_boxes = self._split_large_table_region(
                        (x, y, x + bw, y + bh), detect_horizontal, detect_vertical, intersection
                    )
                    boxes.extend(split_boxes)
                else:
                    boxes.append((x, y, x + bw, y + bh))
            
            return boxes
        except Exception:
            return []

    def _split_large_table_region(self, bbox: Tuple[int, int, int, int], 
                                 h_lines: np.ndarray, v_lines: np.ndarray, 
                                 intersections: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """큰 테이블 영역을 여러 개의 작은 표로 분할"""
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        
        # ROI 추출
        roi_h = h_lines[y0:y1, x0:x1]
        roi_v = v_lines[y0:y1, x0:x1]
        roi_intersect = intersections[y0:y1, x0:x1]
        
        split_boxes = []
        
        try:
            # 수직 분할선 찾기 (중앙 근처의 강한 수직선)
            v_profile = np.sum(roi_v, axis=0)  # 각 x 좌표별 수직선 강도
            center_x = w // 2
            search_range = w // 4
            
            # 중앙 ±25% 범위에서 가장 강한 수직선 찾기
            start_x = max(0, center_x - search_range)
            end_x = min(w, center_x + search_range)
            
            if end_x > start_x:
                center_region = v_profile[start_x:end_x]
                if len(center_region) > 0 and np.max(center_region) > 0:
                    local_max_idx = np.argmax(center_region)
                    split_x = start_x + local_max_idx
                    
                    # 분할선이 충분히 길고 교차점이 있는지 확인
                    split_line_strength = np.sum(roi_v[:, split_x])
                    split_intersections = np.sum(roi_intersect[:, split_x])
                    
                    if split_line_strength > h * 0.3 and split_intersections > 2:
                        # 좌측 표
                        left_box = (x0, y0, x0 + split_x, y1)
                        # 우측 표  
                        right_box = (x0 + split_x, y0, x1, y1)
                        
                        # 각 분할된 영역이 최소 크기를 만족하는지 확인
                        left_w, left_h = split_x, h
                        right_w, right_h = w - split_x, h
                        
                        if (left_w > 50 and left_h > 50 and left_w * left_h > 5000 and
                            right_w > 50 and right_h > 50 and right_w * right_h > 5000):
                            split_boxes.extend([left_box, right_box])
                            return split_boxes
            
            # 분할 실패 시 원본 반환
            split_boxes.append(bbox)
            
        except Exception:
            # 오류 시 원본 반환
            split_boxes.append(bbox)
        
        return split_boxes

    def _is_likely_table(self, x: int, y: int, w: int, h: int, 
                        h_lines: np.ndarray, v_lines: np.ndarray, 
                        intersections: np.ndarray) -> bool:
        """영역이 표일 가능성을 종합 평가"""
        try:
            # ROI 추출
            roi_h = h_lines[y:y+h, x:x+w]
            roi_v = v_lines[y:y+h, x:x+w]
            roi_intersect = intersections[y:y+h, x:x+w]
            
            # 1. 격자 패턴 검사 (수평선과 수직선의 균형)
            h_line_count = np.sum(roi_h > 0)
            v_line_count = np.sum(roi_v > 0)
            total_pixels = w * h
            
            h_line_ratio = h_line_count / total_pixels if total_pixels > 0 else 0
            v_line_ratio = v_line_count / total_pixels if total_pixels > 0 else 0
            
            # 수평선과 수직선이 모두 적절히 있어야 함
            if h_line_ratio < 0.01 or v_line_ratio < 0.01:
                return False
            
            # 2. 교차점 분포 검사 (규칙적인 격자인지)
            intersection_count = np.sum(roi_intersect > 0)
            intersection_density = intersection_count / total_pixels if total_pixels > 0 else 0
            
            # 교차점이 너무 적으면 단순 박스/다이어그램
            if intersection_density < 0.0002:
                return False
            
            # 3. 선분의 연속성 검사 (표는 연속된 선이 많음)
            # 수평선 연속성
            h_profile = np.sum(roi_h, axis=1)  # 각 행별 수평선 강도
            h_continuous_rows = np.sum(h_profile > w * 0.1)  # 너비의 10% 이상 선이 있는 행
            h_continuity = h_continuous_rows / h if h > 0 else 0
            
            # 수직선 연속성  
            v_profile = np.sum(roi_v, axis=0)  # 각 열별 수직선 강도
            v_continuous_cols = np.sum(v_profile > h * 0.1)  # 높이의 10% 이상 선이 있는 열
            v_continuity = v_continuous_cols / w if w > 0 else 0
            
            # 4. 종합 점수 계산
            grid_score = min(h_line_ratio, v_line_ratio) * 100  # 균형잡힌 격자
            intersection_score = intersection_density * 10000  # 교차점 밀도
            continuity_score = (h_continuity + v_continuity) / 2 * 10  # 선분 연속성
            
            total_score = grid_score + intersection_score + continuity_score
            
            # 5. 임계값 기준 판단
            # 표: 격자 패턴 + 교차점 + 연속성 모두 양호
            # 다이어그램: 불규칙한 선분, 교차점 부족, 연속성 낮음
            return total_score > 1.5  # 경험적 임계값
            
        except Exception:
            return True  # 오류 시 보수적으로 표로 간주

    def _validate_table_content(self, image_np: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """텍스트 패턴 기반으로 표 내용 검증"""
        try:
            # ROI 추출
            roi = image_np[y:y+h, x:x+w]
            if roi.size == 0:
                return False
            
            # OCR로 텍스트 추출 (Tesseract 사용)
            if not PYTESSERACT_AVAILABLE:
                logger.warning("Tesseract OCR이 사용 불가능합니다. 텍스트 검증을 건너뜁니다.")
                return True  # OCR 없으면 통과
            
            try:
                # 이미지 전처리 (OCR 정확도 향상)
                if len(roi.shape) == 3:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                else:
                    gray_roi = roi
                
                # 이진화로 텍스트 선명하게
                _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # OCR 실행 (한글+영어)
                ocr_text = pytesseract.image_to_string(
                    binary_roi, 
                    lang='kor+eng',
                    config='--psm 6'  # 단일 텍스트 블록으로 처리
                ).strip()
                
                if not ocr_text:
                    return False  # 텍스트가 없으면 표가 아님
                
                # 표 특징적 텍스트 패턴 정의
                table_patterns = [
                    # 감쇠력 관련 패턴
                    '하-약', '하-중', '하-강',
                    '중-약', '중-중', '중-강',
                    '상-약', '상-중', '상-강',
                    # 음식 관련 패턴
                    '잔', '그릇', '접시',
                    '음료', '국', '탕',
                    '없음', '이상',
                    # 숫자 패턴
                    '1-2', '2-3', '3-4', '4-5',
                    # 일반적인 표 패턴
                    '구분', '항목', '내용', '값',
                    '분류', '종류', '유형'
                ]
                
                # 3D 다이어그램/기술 문서 특징적 패턴 (제외 대상)
                diagram_patterns = [
                    '카메라', '센서', '활용',
                    '3D', '모델', '렌더링',
                    '투명', '반투명', '그라데이션',
                    '레이어', '층', '단계',
                    'Plot', 'Diagram', 'Model',
                    'SAMSUNG', 'Research',  # 브랜드명도 제외
                    '감쇠력', '조절', '판단'  # 제목도 제외 (표 내용이 아님)
                ]
                
                # 패턴 매칭 점수 계산
                table_score = 0
                diagram_score = 0
                
                ocr_lower = ocr_text.lower()
                
                for pattern in table_patterns:
                    if pattern in ocr_text or pattern.lower() in ocr_lower:
                        table_score += 1
                
                for pattern in diagram_patterns:
                    if pattern in ocr_text or pattern.lower() in ocr_lower:
                        diagram_score += 1
                
                # 판단 로직 (더 엄격하게)
                if diagram_score > 0:  # 다이어그램 패턴이 하나라도 있으면 제외
                    logger.info(f"다이어그램으로 판단: diagram_score={diagram_score}, table_score={table_score}, text='{ocr_text[:50]}'")
                    return False
                
                if table_score >= 3:  # 표 패턴이 3개 이상 있어야 표로 판단 (더 엄격)
                    logger.info(f"표로 판단: table_score={table_score}, text='{ocr_text[:50]}'")
                    return True
                
                # 패턴이 부족하면 의심
                if table_score < 2:
                    logger.info(f"패턴 부족으로 제외: table_score={table_score}, text='{ocr_text[:50]}'")
                    return False
                
                # 텍스트가 너무 적거나 패턴이 없으면 의심
                if len(ocr_text.strip()) < 5:
                    logger.debug(f"텍스트 부족으로 제외: text='{ocr_text}'")
                    return False
                
                # 기본적으로 통과 (보수적)
                return True
                
            except Exception as e:
                logger.debug(f"OCR 실행 오류: {e}")
                return True  # OCR 오류 시 보수적으로 통과
                
        except Exception as e:
            logger.debug(f"텍스트 검증 오류: {e}")
            return True  # 전체 오류 시 보수적으로 통과


    def _detect_candidate_regions_simple(self, image_np: np.ndarray) -> List[List[float]]:
        """개선된 후보 영역 검출 (표 분리 검출 포함)"""
        try:
            if image_np is None or image_np.size == 0:
                return []
            
            # 그레이스케일 변환
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            h, w = gray.shape
            
            # 1. 표 구조 검출 (수평/수직선 기반)
            table_candidates = self._detect_table_regions(gray)
            
            # 2. 일반 윤곽선 검출 (그림/다이어그램용)
            figure_candidates = self._detect_figure_regions(gray)
            
            # 3. 결합 및 중복 제거
            all_candidates = table_candidates + figure_candidates
            filtered_candidates = self._remove_overlapping_candidates(all_candidates)
            
            logger.info(f"후보 영역 검출: 표 {len(table_candidates)}개, 그림 {len(figure_candidates)}개 → 최종 {len(filtered_candidates)}개")
            return filtered_candidates
            
        except Exception as e:
            logger.warning(f"후보 영역 검출 실패: {e}")
            return []
    
    def _detect_table_regions(self, gray: np.ndarray) -> List[List[float]]:
        """표 영역 전용 검출 (격자 구조 기반)"""
        h, w = gray.shape
        candidates = []
        
        try:
            # 이진화
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 수평/수직선 검출
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//30), 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h//30)))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # 격자 구조 생성
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # 연결된 구성요소 찾기
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                area = bw * bh
                
                # 표 후보 필터링
                if area < 8000:  # 최소 면적
                    continue
                if min(bw, bh) < 50:  # 최소 변 길이
                    continue
                if max(bw, bh) / min(bw, bh) > 5:  # 종횡비 제한
                    continue
                
                # 격자 밀도 확인
                roi_mask = table_mask[y:y+bh, x:x+bw]
                grid_density = cv2.countNonZero(roi_mask) / area if area > 0 else 0
                
                if grid_density < 0.01:  # 격자 밀도가 너무 낮으면 제외
                    continue
                
                # 큰 영역은 분할 시도
                if bw > w * 0.5 and bh > h * 0.2:
                    split_boxes = self._split_table_region(x, y, bw, bh, horizontal_lines, vertical_lines)
                    for sx, sy, sw, sh in split_boxes:
                        norm_bbox = [sx/w, sy/h, (sx+sw)/w, (sy+sh)/h]
                        candidates.append(norm_bbox)
                else:
                    norm_bbox = [x/w, y/h, (x+bw)/w, (y+bh)/h]
                    candidates.append(norm_bbox)
            
        except Exception as e:
            logger.warning(f"표 영역 검출 실패: {e}")
        
        return candidates
    
    def _detect_figure_regions(self, gray: np.ndarray) -> List[List[float]]:
        """그림/다이어그램 영역 검출"""
        h, w = gray.shape
        candidates = []
        
        try:
            # 이진화
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                area = bw * bh
                
                # 그림 후보 필터링
                if area < 3000:  # 최소 면적 (표보다 작게)
                    continue
                if min(bw, bh) < 30:  # 최소 변 길이
                    continue
                if max(bw, bh) / min(bw, bh) > 10:  # 종횡비 제한
                    continue
                
                norm_bbox = [x/w, y/h, (x+bw)/w, (y+bh)/h]
                candidates.append(norm_bbox)
            
        except Exception as e:
            logger.warning(f"그림 영역 검출 실패: {e}")
        
        return candidates
    
    def _split_table_region(self, x: int, y: int, w: int, h: int, 
                           h_lines: np.ndarray, v_lines: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """큰 표 영역을 여러 표로 분할"""
        try:
            # ROI 추출
            roi_h = h_lines[y:y+h, x:x+w]
            roi_v = v_lines[y:y+h, x:x+w]
            
            # 수직 분할선 찾기 (중앙 근처)
            v_profile = np.sum(roi_v, axis=0)
            center_x = w // 2
            search_range = w // 4
            
            start_x = max(0, center_x - search_range)
            end_x = min(w, center_x + search_range)
            
            if end_x > start_x:
                center_region = v_profile[start_x:end_x]
                if len(center_region) > 0 and np.max(center_region) > h * 0.3:
                    local_max_idx = np.argmax(center_region)
                    split_x = start_x + local_max_idx
                    
                    # 좌우로 분할
                    left_box = (x, y, split_x, h)
                    right_box = (x + split_x, y, w - split_x, h)
                    
                    # 최소 크기 확인
                    if (split_x > 50 and (w - split_x) > 50 and 
                        split_x * h > 5000 and (w - split_x) * h > 5000):
                        return [left_box, right_box]
            
            # 분할 실패 시 원본 반환
            return [(x, y, w, h)]
            
        except Exception:
            return [(x, y, w, h)]
    
    def _remove_overlapping_candidates(self, candidates: List[List[float]]) -> List[List[float]]:
        """중복되는 후보 영역 제거"""
        if len(candidates) <= 1:
            return candidates
        
        # IoU 기반 중복 제거
        filtered = []
        for i, bbox1 in enumerate(candidates):
            is_duplicate = False
            for j, bbox2 in enumerate(filtered):
                iou = self._calculate_iou(bbox1, bbox2)
                if iou > 0.5:  # 50% 이상 겹치면 중복으로 간주
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(bbox1)
        
        return filtered
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """두 bbox의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    extractor = AdvancedLayoutExtractor()
    
    # PDF 처리
    pdf_path = "pdf/음료흘림방지기술.pdf"
    result = extractor.extract_hierarchical_layout(pdf_path)
    
    # 결과 저장
    with open("advanced_layout_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("고도화된 레이아웃 추출 완료!")
