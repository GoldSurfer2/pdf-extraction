#!/usr/bin/env python3
"""
하이브리드 문서 처리기: Docling (텍스트 보존) + GPT-4o-mini (시각 분석)
"""

import logging
import json
import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Any
import base64
from io import BytesIO

# Docling 관련
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

# GPT-4o-mini 관련
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# PDF 이미지 변환
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# 고도화 레이아웃 추출기(시각 레이아웃 보존)
try:
    from advanced_layout_extractor import AdvancedLayoutExtractor
    ADV_LAYOUT_AVAILABLE = True
except Exception:
    AdvancedLayoutExtractor = None
    ADV_LAYOUT_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDoclingGPT4Processor:
    """
    Docling으로 텍스트 보존, GPT-4o-mini로 시각요소 분석하는 하이브리드 처리기
    """
    
    def __init__(self):
        """하이브리드 처리기 초기화"""
        load_dotenv()
        
        # GPT-4o-mini 초기화
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.system_message = SystemMessage("""
            Extract and analyze ALL visual elements from the provided document image comprehensively.
            You are an expert in multimodal document understanding specializing in various document types.

            IMPORTANT: Analyze EVERY visual element in the image including:

            1. **MATHEMATICAL FORMULAS & EQUATIONS**:
               - Convert all mathematical expressions to precise LaTeX format
               - Extract variable definitions, coefficients, and constants
               - Include units, physical meanings, and relationships
               
            2. **TABLES & DATA**:
               - Extract all tabular data with accurate values
               - Preserve table structure (rows, columns, headers)
               - Include table captions and footnotes
               - Convert to structured format (markdown tables or JSON)
               
            3. **GRAPHS & CHARTS**:
               - Describe graph type (line graph, bar chart, scatter plot, etc.)
               - Extract axis labels, units, and scale ranges
               - Identify all data series/lines with their colors and patterns
               - Extract key data points, trends, and relationships
               - Include graph titles and legends
               
            4. **FIGURES & DIAGRAMS**:
               - Describe schematic diagrams, flowcharts, illustrations
               - Identify components, connections, and relationships
               - Extract labels, annotations, and captions
               - Explain the purpose and meaning of the diagram
               
            5. **IMAGES & PHOTOGRAPHS**:
               - Describe experimental setups, microscopy images, photographs
               - Identify important features, measurements, scale bars
               - Include image captions and context

            FORMAT YOUR RESPONSE AS:
            ## VISUAL CONTENT ANALYSIS

            ### Mathematical Formulas
            [List all formulas with LaTeX conversion and detailed explanations]

            ### Tables and Data
            [Extract all table content with proper structure]

            ### Graphs and Charts
            [Analyze all charts with data extraction and trend analysis]

            ### Figures and Diagrams
            [Describe all diagrams and visual illustrations]

            ### Additional Visual Elements
            [Any other visual content not covered above]

            Be thorough and extract maximum information for multimodal RAG applications.
        """)
        
        # GPU 환경 설정 (멀티GPU NCCL 에러 방지)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 1번 사용 (0번은 다른 작업 진행 중)
        
        # Docling 초기화 (텍스트 추출 최적화 포함)
        self.pipeline_options = PdfPipelineOptions(
            do_table_structure=True,           # 표 구조 분석 활성화
            do_ocr=True,                      # OCR 텍스트 인식 활성화 (중요!)
            images_scale=2.0,                 # 고해상도 이미지
            generate_page_images=True,        # 페이지 이미지 생성
            generate_table_images=True,       # 표 이미지 생성  
            generate_picture_images=True      # 그림 이미지 생성
        )
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        logger.info("하이브리드 Docling + GPT-4o-mini 처리기 초기화 완료")
        # 레이아웃 보존 추출기 준비(선택적)
        self.adv_extractor = AdvancedLayoutExtractor() if ADV_LAYOUT_AVAILABLE else None
    
    def extract_text_structure_with_docling(self, pdf_path: str) -> Dict[str, Any]:
        """
        Docling으로 텍스트와 구조 추출 (원문 보존)
        """
        logger.info(f"Docling으로 텍스트 구조 추출 시작: {pdf_path}")
        
        try:
            # PDF 변환
            conv_result = self.doc_converter.convert(pdf_path)
            
            # 마크다운 텍스트 추출 (원문 보존)
            markdown_text = conv_result.document.export_to_markdown()
            
            # 표와 이미지 위치 정보 (페이지별 순회로 페이지 번호 확정)
            tables = []
            images = []
            
            pages = getattr(conv_result.document, 'pages', [])
            if pages:
                for page_idx, page in enumerate(pages):
                    try:
                        iterator = page.iterate_items()
                    except Exception:
                        iterator = []
                    for element, level in iterator:
                        if isinstance(element, TableItem):
                            text_val = getattr(element, 'text', '') if hasattr(element, 'text') else ''
                            tables.append({
                                "text": text_val,
                                "page": page_idx + 1
                            })
                        elif isinstance(element, PictureItem):
                            caption_val = self._safe_get_caption_text(element)
                            images.append({
                                "page": page_idx + 1,
                                "caption": caption_val
                            })
            else:
                # 페이지 API가 없을 때의 안전한 폴백 (기존 방식)
                for element, level in conv_result.document.iterate_items():
                    if isinstance(element, TableItem):
                        page_num = self._safe_get_page_number(element)
                        text_val = getattr(element, 'text', '') if hasattr(element, 'text') else ''
                        tables.append({
                            "text": text_val,
                            "page": page_num
                        })
                    elif isinstance(element, PictureItem):
                        page_num = self._safe_get_page_number(element)
                        caption_val = self._safe_get_caption_text(element)
                        images.append({
                            "page": page_num,
                            "caption": caption_val
                        })
            
            result = {
                "markdown_text": markdown_text,
                "tables": tables,
                "image_locations": images,
                "total_pages": len(conv_result.document.pages)
            }
            
            logger.info(f"Docling 추출 완료 - 페이지: {result['total_pages']}, 표: {len(tables)}, 이미지: {len(images)}")
            return result
            
        except Exception as e:
            logger.error(f"Docling 텍스트 추출 실패: {e}")
            raise

    def _safe_get_page_number(self, element) -> int:
        """요소로부터 페이지 번호를 안전하게 추출 (가능하면 1-based)"""
        candidates = [
            ('page_number', False),  # 1-based일 수 있음
            ('page_idx', True),      # 0-based 가능성 높음
            ('page_index', True),
            ('page', False),
            ('page_no', False)
        ]
        for attr, needs_plus_one in candidates:
            if hasattr(element, attr):
                try:
                    val = getattr(element, attr)
                    if isinstance(val, (int, float)):
                        page = int(val)
                    else:
                        page = int(str(val))
                    if needs_plus_one:
                        page = page + 1
                    # 0 또는 음수 방지
                    if page <= 0:
                        page = 1
                    return page
                except Exception:
                    continue
        # PictureItem의 경우 image_ref 내부에 page_index가 존재할 수 있음
        try:
            image_ref = getattr(element, 'image_ref', None)
            if image_ref is not None:
                # page_index 또는 page_idx 우선 사용
                if hasattr(image_ref, 'page_index'):
                    idx = int(getattr(image_ref, 'page_index'))
                    return max(1, idx + 1)
                if hasattr(image_ref, 'page_idx'):
                    idx = int(getattr(image_ref, 'page_idx'))
                    return max(1, idx + 1)
                # 일부 케이스: image_ref가 dict 유사
                if isinstance(image_ref, dict):
                    if 'page_index' in image_ref:
                        idx = int(image_ref['page_index'])
                        return max(1, idx + 1)
                    if 'page_idx' in image_ref:
                        idx = int(image_ref['page_idx'])
                        return max(1, idx + 1)
        except Exception:
            pass
        return 1

    def _safe_get_caption_text(self, element) -> str:
        """PictureItem의 캡션 텍스트 안전 추출"""
        if not hasattr(element, 'caption'):
            return ""
        cap = getattr(element, 'caption')
        try:
            if cap is None:
                return ""
            if isinstance(cap, dict):
                return cap.get('text', '') or cap.get('caption', '') or ''
            # 객체에 text 속성이 있는 경우
            if hasattr(cap, 'text'):
                return getattr(cap, 'text') or ""
            if isinstance(cap, str):
                return cap
        except Exception:
            return ""
        return ""
    
    def extract_all_visuals_with_gpt4(self, pdf_path: str, max_pages: int = None) -> Dict[str, Any]:
        """
        GPT-4o-mini로 모든 시각요소 분석
        """
        logger.info(f"GPT-4o-mini로 전체 시각요소 분석 시작: {pdf_path}")
        
        try:
            # PDF를 페이지별 이미지로 변환
            doc = fitz.open(pdf_path)
            visual_analyses = []
            
            pages_to_process = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
            logger.info(f"처리할 페이지 수: {pages_to_process} / {doc.page_count}")
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # 고해상도 이미지 변환
                mat = fitz.Matrix(2.5, 2.5)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # GPT-4o-mini로 시각요소 분석
                analysis = self._analyze_all_visuals(image, page_num + 1)
                
                if analysis and analysis.strip():
                    visual_analyses.append({
                        "page": page_num + 1,
                        "visual_analysis": analysis,
                        "has_visuals": True
                    })
                    logger.info(f"페이지 {page_num + 1}: 시각요소 분석 완료")
                else:
                    visual_analyses.append({
                        "page": page_num + 1,
                        "visual_analysis": "",
                        "has_visuals": False
                    })
                    logger.info(f"페이지 {page_num + 1}: 시각요소 없음")
            
            doc.close()
            
            return {
                "visual_analyses": visual_analyses,
                "total_pages_analyzed": pages_to_process
            }
            
        except Exception as e:
            logger.error(f"GPT-4o-mini 시각 분석 실패: {e}")
            raise

    def extract_layout_preserved(self, pdf_path: str) -> Dict[str, Any]:
        """AdvancedLayoutExtractor를 이용해 레이아웃 보존 요소를 추출하고 직렬화"""
        if not self.adv_extractor:
            return {"tables": [], "figures": [], "elements": [], "method": "disabled"}
        try:
            # AI 기반 레이아웃 추출 사용
            if hasattr(self.adv_extractor, '_extract_with_ai_classification'):
                layout = self.adv_extractor._extract_with_ai_classification(pdf_path)
            else:
                # 폴백: 기존 방식
                layout = self.adv_extractor._extract_with_visual_analysis(pdf_path)
            pages = layout.get("pages", [])
            tables: List[Dict[str, Any]] = []
            figures: List[Dict[str, Any]] = []
            elements: List[Dict[str, Any]] = []
            for page_item in pages:
                page_idx = int(page_item.get("page", 0))  # 0-based 저장됨
                page_num = page_idx + 1
                for el in page_item.get("elements", []):
                    # el은 dataclass LayoutElement일 가능성이 높음 → getattr 사용
                    etype = getattr(el, "element_type", None)
                    etype_val = getattr(etype, "value", str(etype)) if etype is not None else "unknown"
                    bbox = getattr(el, "bbox", None)
                    content = getattr(el, "content", "")
                    conf = getattr(el, "confidence", 0.0)
                    serialized = {
                        "type": etype_val,
                        "page": page_num,
                        "bbox": bbox,
                        "content": content,
                        "confidence": conf
                    }
                    elements.append(serialized)
                    if etype_val == "table":
                        tables.append({
                            "page": page_num,
                            "bbox": bbox,
                            "caption": "",
                            "confidence": conf
                        })
                    elif etype_val == "figure":
                        figures.append({
                            "page": page_num,
                            "bbox": bbox,
                            "caption": content or "",
                            "confidence": conf
                        })
            return {"tables": tables, "figures": figures, "elements": elements, "method": layout.get("method", "visual_pymupdf")}
        except Exception as e:
            logger.warning(f"레이아웃 보존 추출 실패: {e}")
            return {"tables": [], "figures": [], "elements": [], "method": "error"}

    def _merge_layout_into_docling(self, docling_result: Dict[str, Any], layout_data: Dict[str, Any]) -> None:
        """고도화 레이아웃의 표/그림을 Docling 결과에 병합 + 교차검증 필터링"""
        if not docling_result:
            return
        
        # Docling에서 감지한 페이지별 표/이미지 개수 수집
        docling_table_pages = set()
        docling_image_pages = set()
        
        for t in docling_result.get("tables", []):
            docling_table_pages.add(int(t.get("page", 1)))
        for img in docling_result.get("image_locations", []):
            docling_image_pages.add(int(img.get("page", 1)))
        
        dl_tables = docling_result.get("tables", [])
        dl_images = docling_result.get("image_locations", [])
        
        # 교차검증: Docling이 표를 감지한 페이지의 레이아웃 표만 추가
        validated_tables = 0
        for t in layout_data.get("tables", []):
            page = int(t.get("page", 1))
            confidence = float(t.get("confidence", 0.65))
            
            # Docling이 같은 페이지에서 표를 감지했거나, confidence가 높으면 허용
            if page in docling_table_pages or confidence > 0.8:
                dl_tables.append({
                    "text": "",
                    "page": page,
                    "bbox": t.get("bbox", []),
                    "confidence": confidence,
                    "source": "layout_validated"
                })
                validated_tables += 1
        
        # 그림은 더 관대하게 (Docling이 놓치는 경우가 많음)
        validated_figures = 0
        for f in layout_data.get("figures", []):
            page = int(f.get("page", 1))
            confidence = float(f.get("confidence", 0.70))
            
            # confidence 0.6 이상이면 추가
            if confidence >= 0.6:
                dl_images.append({
                    "page": page,
                    "caption": f.get("caption", ""),
                    "bbox": f.get("bbox", []),
                    "confidence": confidence,
                    "source": "layout_detected"
                })
                validated_figures += 1
        
        docling_result["tables"] = dl_tables
        docling_result["image_locations"] = dl_images
        
        logger.info(f"레이아웃 병합 완료: 표 {validated_tables}개, 그림 {validated_figures}개 추가")
    
    def _analyze_all_visuals(self, image: Image.Image, page_num: int) -> str:
        """
        개별 페이지의 모든 시각요소 분석
        """
        try:
            # 이미지를 base64로 인코딩
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 종합적 시각요소 분석을 위한 프롬프트
            text_content = f"""Here is the context for comprehensive visual analysis:
                
                ### Page Context:
                Page {page_num} from document processing
                
                Please perform a complete and detailed analysis of ALL visual elements in this image.
                This analysis will be used for a multimodal RAG system, so accuracy and completeness are critical.
                
                Extract and describe every table, graph, chart, formula, diagram, and figure you can identify.
                Do not miss any visual information that could be valuable for question-answering.
                
                ### COMPREHENSIVE VISUAL ANALYSIS:"""
            
            # 메시지 구성
            message = HumanMessage(
                content=[
                    {"type": "text", "text": text_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            )
            
            # GPT-4o-mini 호출
            response = self.model.invoke([self.system_message, message])
            
            return response.content
            
        except Exception as e:
            logger.error(f"페이지 {page_num} 시각 분석 실패: {e}")
            return ""
    
    def process_hybrid(self, pdf_path: str, max_pages: int = None, doc_pages: int = None, selected_pages: List[int] | None = None) -> Dict[str, Any]:
        """
        하이브리드 처리: Docling + GPT-4o-mini 결합
        """
        logger.info(f"하이브리드 처리 시작: {pdf_path}")
        
        try:
            # 필요 시 앞쪽 N페이지만 포함한 임시 PDF 생성
            input_pdf_path = pdf_path
            temp_pdf_path = None
            if selected_pages:
                temp_pdf_path = self._create_temp_pdf_selected_pages(pdf_path, selected_pages)
                input_pdf_path = temp_pdf_path
                logger.info(f"임시 PDF 생성(선택 페이지 {selected_pages}): {input_pdf_path}")
            elif doc_pages is not None and doc_pages > 0:
                temp_pdf_path = self._create_temp_pdf_first_pages(pdf_path, doc_pages)
                input_pdf_path = temp_pdf_path
                logger.info(f"임시 PDF 생성(앞 {doc_pages}페이지): {input_pdf_path}")

            # 1단계: Docling으로 텍스트 구조 추출
            docling_result = self.extract_text_structure_with_docling(input_pdf_path)
            
            # 1.5단계: 레이아웃 보존 요소 추가 추출(선택적)
            layout_preserved = self.extract_layout_preserved(input_pdf_path)
            # Docling 결과에 표/이미지 병합
            self._merge_layout_into_docling(docling_result, layout_preserved)

            # 2단계: GPT-4o-mini로 모든 시각요소 분석
            gpt4_result = self.extract_all_visuals_with_gpt4(input_pdf_path, max_pages)
            
            # 3단계: 결합
            combined_result = {
                "source_pdf": pdf_path,
                "processing_method": "Hybrid: Docling (text preservation) + GPT-4o-mini (visual analysis)",
                "docling_extraction": docling_result,
                "gpt4_visual_analysis": gpt4_result,
                "layout_preserved": layout_preserved,
                "summary": {
                    "total_pages": docling_result["total_pages"],
                    "pages_with_visuals": sum(1 for p in gpt4_result["visual_analyses"] if p["has_visuals"]),
                    "tables_found": len(docling_result["tables"]),
                    "images_found": len(docling_result["image_locations"])
                }
            }
            
            logger.info(f"하이브리드 처리 완료:")
            logger.info(f"  - 총 페이지: {combined_result['summary']['total_pages']}")
            logger.info(f"  - 시각요소 있는 페이지: {combined_result['summary']['pages_with_visuals']}")
            logger.info(f"  - 표: {combined_result['summary']['tables_found']}개")
            logger.info(f"  - 이미지: {combined_result['summary']['images_found']}개")
            
            # 임시 파일 정리
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except Exception:
                    pass

            return combined_result
            
        except Exception as e:
            logger.error(f"하이브리드 처리 실패: {e}")
            raise

    def _create_temp_pdf_first_pages(self, pdf_path: str, num_pages: int) -> str:
        """원본 PDF의 앞쪽 N페이지만 포함하는 임시 PDF 생성 후 경로 반환"""
        temp_name = f"_tmp_first_{num_pages}_{Path(pdf_path).stem}.pdf"
        temp_path = str(Path(pdf_path).parent / temp_name)
        src = None
        dst = None
        try:
            src = fitz.open(pdf_path)
            dst = fitz.open()
            pages_to_copy = min(max(1, num_pages), src.page_count)
            for i in range(pages_to_copy):
                dst.insert_pdf(src, from_page=i, to_page=i)
            dst.save(temp_path)
            return temp_path
        finally:
            try:
                if dst is not None:
                    dst.close()
            except Exception:
                pass
            try:
                if src is not None:
                    src.close()
            except Exception:
                pass

    def _create_temp_pdf_selected_pages(self, pdf_path: str, pages: List[int]) -> str:
        """지정된 1-based 페이지 목록만 포함하는 임시 PDF 생성"""
        temp_name = f"_tmp_selected_{'-'.join(map(str, pages))}_{Path(pdf_path).stem}.pdf"
        temp_path = str(Path(pdf_path).parent / temp_name)
        src = None
        dst = None
        try:
            src = fitz.open(pdf_path)
            dst = fitz.open()
            for p in pages:
                idx = max(1, int(p)) - 1
                if 0 <= idx < src.page_count:
                    dst.insert_pdf(src, from_page=idx, to_page=idx)
            dst.save(temp_path)
            return temp_path
        finally:
            try:
                if dst is not None:
                    dst.close()
            except Exception:
                pass
            try:
                if src is not None:
                    src.close()
            except Exception:
                pass

    def create_debug_overlay(self, pdf_path: str, result: Dict[str, Any], debug_dir: str) -> None:
        """디버그 오버레이 이미지 생성 - 검출된 표/그림 bbox를 시각화"""
        try:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            
            doc = fitz.open(pdf_path)
            layout_data = result.get("layout_preserved", {})
            
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                page_num = page_idx + 1
                
                # 고해상도 이미지 생성
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(BytesIO(img_data))
                
                # 오버레이 그리기
                draw = ImageDraw.Draw(image)
                img_w, img_h = image.size
                
                # 해당 페이지의 요소들 필터링
                page_elements = [e for e in layout_data.get("elements", []) 
                               if e.get("page") == page_num]
                
                for element in page_elements:
                    bbox = element.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    
                    # 정규화된 좌표를 픽셀 좌표로 변환
                    x0, y0, x1, y1 = bbox
                    px0 = int(x0 * img_w)
                    py0 = int(y0 * img_h)
                    px1 = int(x1 * img_w)
                    py1 = int(y1 * img_h)
                    
                    # 타입별 색상
                    element_type = element.get("type", "unknown")
                    colors = {
                        "table": "red",
                        "figure": "blue", 
                        "title": "green",
                        "paragraph": "orange"
                    }
                    color = colors.get(element_type, "gray")
                    
                    # 박스 그리기
                    draw.rectangle([px0, py0, px1, py1], outline=color, width=3)
                    
                    # 라벨 추가
                    confidence = element.get("confidence", 0.0)
                    label = f"{element_type} ({confidence:.2f})"
                    
                    try:
                        # 기본 폰트 사용
                        draw.text((px0, py0-20), label, fill=color)
                    except Exception:
                        # 폰트 로드 실패 시 라벨 생략
                        pass
                
                # 저장
                output_path = os.path.join(debug_dir, f"page_{page_num:03d}_overlay.png")
                image.save(output_path)
                logger.info(f"디버그 오버레이 저장: {output_path}")
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"디버그 오버레이 생성 실패: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="하이브리드 Docling + GPT-4o-mini 문서 처리기")
    parser.add_argument("--pdf", type=str, required=True, help="처리할 PDF 파일 경로")
    parser.add_argument("--output", type=str, default="results/beverage_tech_result.json", help="결과 저장 파일명")
    parser.add_argument("--max-pages", type=int, default=None, help="GPT-4o-mini로 분석할 최대 페이지 수 (None이면 전체 페이지)")
    parser.add_argument("--page", type=int, default=None, help="특정 1-based 페이지만 처리 (Docling+GPT 대상)")
    parser.add_argument("--timestamp", action="store_true", help="출력 파일명에 생성 시각 접미사 추가")
    parser.add_argument("--debug-overlay", type=str, default=None, help="디버그 오버레이 이미지 저장 디렉토리")
    parser.add_argument("--doc-pages", type=int, default=None, help="Docling/GPT 모두 앞쪽 N페이지만 처리")
    
    args = parser.parse_args()
    
    try:
        # 하이브리드 처리기 초기화
        processor = HybridDoclingGPT4Processor()
        
        # 하이브리드 처리 실행
        selected_pages = [args.page] if args.page else None
        result = processor.process_hybrid(args.pdf, args.max_pages, args.doc_pages, selected_pages)
        
        # 결과 저장(타임스탬프 옵션)
        output_path = args.output
        if args.timestamp:
            stem = Path(output_path).stem
            suffix = Path(output_path).suffix or ".json"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Path(output_path).with_name(f"{stem}_{ts}{suffix}"))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 디버그 오버레이 생성
        if args.debug_overlay:
            processor.create_debug_overlay(args.pdf, result, args.debug_overlay)
        
        logger.info(f"하이브리드 처리 결과가 {output_path}에 저장되었습니다")
        
        # 요약 출력
        print(f"\n=== 하이브리드 처리 결과 ===")
        print(f"PDF: {result['source_pdf']}")
        print(f"처리 방식: {result['processing_method']}")
        print(f"총 페이지: {result['summary']['total_pages']}")
        print(f"복잡한 시각요소 페이지: {result['summary']['pages_with_visuals']}")
        print(f"표: {result['summary']['tables_found']}개")
        print(f"이미지: {result['summary']['images_found']}개")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
