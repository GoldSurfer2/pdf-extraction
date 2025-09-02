#!/usr/bin/env python3
"""
GPT-4o-mini를 사용한 종합적 시각요소 추출 시스템
- 수식, 표, 그래프, 차트, 다이어그램 등 모든 시각요소 처리
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import fitz  # PyMuPDF
import base64
from io import BytesIO

# LangChain 관련 import
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"필요한 라이브러리를 찾을 수 없습니다: {e}")
    print("pip install langchain-openai python-dotenv 로 설치해주세요")
    exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4ComprehensiveVisualProcessor:
    def __init__(self):
        """
        GPT-4o-mini 종합적 시각요소 처리기 초기화
        """
        # 환경 변수 로드
        load_dotenv()
        
        try:
            logger.info("GPT-4o-mini 종합 시각요소 처리기 초기화 중...")
            
            # GPT-4o-mini 모델 초기화
            self.model = ChatOpenAI(model="gpt-4o-mini")
            
            # 종합적 시각요소 추출을 위한 시스템 메시지
            self.system_message = SystemMessage("""
                Extract and analyze ALL visual elements from the provided scientific paper image comprehensively.
                You are an expert in multimodal document understanding specializing in scientific papers.

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
                [List all formulas in LaTeX with explanations]

                ### Tables and Data
                [Extract all tabular information]

                ### Graphs and Charts  
                [Detailed analysis of all graphical content]

                ### Figures and Diagrams
                [Description of diagrams, schematics, illustrations]

                ### Additional Visual Elements
                [Any other visual content not covered above]

                Be extremely thorough and precise. Extract ALL visual information that would be valuable for a multimodal RAG system.
            """)
            
            logger.info("GPT-4o-mini 종합 시각요소 처리기 초기화 완료")
            
        except Exception as e:
            logger.error(f"처리기 초기화 실패: {e}")
            raise
    
    def pdf_to_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDF를 페이지별 고해상도 이미지로 변환
        """
        try:
            doc = fitz.open(pdf_path)
            pages_info = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # 고해상도 변환 (시각요소 정확도를 위해)
                mat = fitz.Matrix(2.5, 2.5)  # 더 높은 해상도
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # PIL 이미지로 변환
                image = Image.open(BytesIO(img_data))
                
                pages_info.append({
                    "page_number": page_num + 1,
                    "image": image,
                    "image_size": [image.width, image.height]
                })
                
                logger.info(f"페이지 {page_num + 1} 변환 완료 ({image.width}x{image.height})")
            
            doc.close()
            return pages_info
            
        except Exception as e:
            logger.error(f"PDF 이미지 변환 실패: {e}")
            raise
    
    def extract_all_visual_elements(self, image: Image.Image, page_context: str = "") -> str:
        """
        GPT-4o-mini로 모든 시각요소 추출
        """
        try:
            # 이미지를 base64로 인코딩
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 종합적 시각요소 분석을 위한 프롬프트
            text_content = f"""Here is the context for comprehensive visual analysis:
                
                ### Page Context:
                {page_context if page_context else "Scientific research paper page containing various visual elements"}
                
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
            logger.error(f"GPT-4o-mini 시각요소 추출 실패: {e}")
            return f"시각요소 추출 실패: {str(e)}"
    
    def analyze_visual_content_types(self, analysis_text: str) -> Dict[str, int]:
        """
        추출된 시각요소 유형별 개수 분석 (정확한 버전)
        """
        import re
        
        content_types = {
            # 수식: Mathematical Formulas 섹션의 번호가 있는 항목만 카운팅
            "formulas": self._count_actual_formulas(analysis_text),
            
            # 표: Tables and Data 섹션의 실제 테이블만 카운팅  
            "tables": self._count_actual_tables(analysis_text),
            
            # 그래프/차트: Graphs and Charts 섹션의 실제 그래프만 카운팅
            "graphs": self._count_actual_graphs(analysis_text),
            
            # 그림/다이어그램: Figures and Diagrams 섹션의 실제 그림만 카운팅
            "figures": self._count_actual_figures(analysis_text),
            
            # 데이터 포인트: 실제 수치 데이터 개수
            "data_points": self._count_data_points(analysis_text)
        }
        return content_types
    
    def _count_actual_formulas(self, text: str) -> int:
        """Mathematical Formulas 섹션의 실제 수식만 카운팅"""
        import re
        
        # "No mathematical formulas" 체크 (명시적으로 없다고 한 경우)
        if "no mathematical formulas" in text.lower() or "does not explicitly contain mathematical formulas" in text.lower():
            return 0
            
        # Mathematical Formulas 섹션 찾기
        formula_section = re.search(r'### Mathematical Formulas(.*?)(?=###|$)', text, re.DOTALL)
        if not formula_section:
            return 0
            
        formula_content = formula_section.group(1)
        
        # 번호가 있는 수식만 카운팅 (1. **Name**, 2. **Name** 형태)
        numbered_formulas = re.findall(r'^\d+\.\s+\*\*[^*]+\*\*', formula_content, re.MULTILINE)
        return len(numbered_formulas)
    
    def _count_actual_tables(self, text: str) -> int:
        """Tables and Data 섹션의 실제 테이블만 카운팅"""
        import re
        
        # 명시적으로 테이블이 없다고 한 경우
        if ("no tables" in text.lower() or 
            "does not present explicit tables" in text.lower() or
            "no explicit tables" in text.lower()):
            return 0
            
        table_section = re.search(r'### Tables and Data(.*?)(?=###|$)', text, re.DOTALL)
        if not table_section:
            return 0
            
        table_content = table_section.group(1)
        
        # 실제 테이블 구조만 카운팅 (마크다운 테이블: |---| 형태)
        table_separators = len(re.findall(r'\|[\s-]*\|[\s-]*\|', table_content))
        if table_separators > 0:
            return table_separators
            
        # 번호가 있는 테이블 항목
        numbered_tables = len(re.findall(r'^\d+\.\s+\*\*.*Table', table_content, re.MULTILINE | re.IGNORECASE))
        return numbered_tables
    
    def _count_actual_graphs(self, text: str) -> int:
        """Graphs and Charts 섹션의 실제 그래프만 카운팅"""
        import re
        
        # 명시적으로 그래프가 없다고 한 경우
        if ("no graphs" in text.lower() or "no charts" in text.lower() or
            "no graphs, charts" in text.lower() or 
            "no graphs or charts" in text.lower()):
            return 0
            
        graph_section = re.search(r'### Graphs and Charts(.*?)(?=###|$)', text, re.DOTALL)
        if not graph_section:
            return 0
            
        graph_content = graph_section.group(1)
        
        # 번호가 있는 그래프/Figure 항목들 카운팅
        numbered_items = re.findall(r'^\d+\.\s+\*\*[^*]+\*\*', graph_content, re.MULTILINE)
        
        # Figure 패턴들 카운팅 (S1(a), S5a, S4 등 모든 형태)
        figure_patterns = re.findall(r'\*\*Figure[^*]*\*\*|\*\*Figures[^*]*\*\*', graph_content)
        
        # 개별 그래프 구분 (S5a and S5b는 2개로 카운팅)
        individual_graphs = 0
        for pattern in figure_patterns:
            if " and " in pattern:
                individual_graphs += 2  # "S5a and S5b" = 2개
            else:
                individual_graphs += 1
        
        return max(len(numbered_items), individual_graphs)
    
    def _count_actual_figures(self, text: str) -> int:
        """Figures and Diagrams 섹션의 실제 그림만 카운팅"""
        import re
        
        # 명시적으로 그림이 없다고 한 경우
        if ("no figures" in text.lower() or "no diagrams" in text.lower() or
            "no traditional diagrams" in text.lower() or
            "no figures or diagrams" in text.lower() or
            "are not explicitly described" in text.lower()):
            return 0
            
        figure_section = re.search(r'### Figures and Diagrams(.*?)(?=###|$)', text, re.DOTALL)
        if not figure_section:
            return 0
            
        figure_content = figure_section.group(1)
        
        # 번호가 있는 figure/diagram 항목들 카운팅
        numbered_figures = re.findall(r'^\d+\.\s+\*\*[^*]+\*\*', figure_content, re.MULTILINE)
        
        # Figure 패턴들 카운팅 (S4, S5 등)
        figure_patterns = re.findall(r'\*\*Figure[^*]*\*\*|\*\*Figures[^*]*\*\*', figure_content)
        
        # 개별 그림 구분
        individual_figures = 0
        for pattern in figure_patterns:
            if " and " in pattern:
                individual_figures += 2  # "S4 (a) and (b)" = 1개 그림 (2개 서브그림)
            else:
                individual_figures += 1
        
        # 실제 시각적 다이어그램을 나타내는 키워드들 (단순 텍스트 설명 제외)
        visual_indicators = [
            "schematic", "diagram shows", "illustration", "drawing", 
            "visual representation", "flowchart", "circuit diagram"
        ]
        
        has_visual_diagram = any(indicator in figure_content.lower() for indicator in visual_indicators)
        
        return max(len(numbered_figures), individual_figures, 1 if has_visual_diagram else 0)
    
    def _count_data_points(self, text: str) -> int:
        """실제 수치 데이터 포인트만 카운팅"""
        import re
        
        # 구체적인 수치 패턴만 카운팅
        numbers = re.findall(r'\d+\.?\d*', text)
        equations = text.count('=')
        percentages = text.count('%')
        
        return len(numbers) + equations + percentages
    
    def process_pdf(self, pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
        """
        PDF에서 종합적 시각요소 추출
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            logger.info(f"GPT-4o-mini 종합 시각요소 추출 시작: {pdf_path}")
            
            # PDF를 이미지로 변환
            pages_info = self.pdf_to_images(pdf_path)
            
            results = {
                "model_used": "GPT-4o-mini",
                "analysis_type": "comprehensive_visual_elements",
                "source_pdf": pdf_path,
                "total_pages": len(pages_info),
                "processed_pages": [],
                "success_count": 0,
                "failure_count": 0,
                "extracted_visual_elements": [],
                "visual_content_summary": {}
            }
            
            # 페이지별 종합 시각요소 추출
            pages_to_process = min(max_pages, len(pages_info))
            logger.info(f"처리할 페이지 수: {pages_to_process} / {len(pages_info)}")
            
            all_content_types = {"formulas": 0, "tables": 0, "graphs": 0, "figures": 0, "data_points": 0}
            
            for i in range(pages_to_process):
                page_info = pages_info[i]
                logger.info(f"페이지 {page_info['page_number']} 종합 시각요소 분석 중...")
                
                try:
                    page_context = f"Page {page_info['page_number']} of scientific research paper - comprehensive visual analysis"
                    visual_analysis = self.extract_all_visual_elements(
                        page_info["image"], 
                        page_context
                    )
                    
                    # 시각요소 유형별 분석
                    content_types = self.analyze_visual_content_types(visual_analysis)
                    
                    # 전체 통계 업데이트
                    for key in all_content_types:
                        all_content_types[key] += content_types.get(key, 0)
                    
                    page_result = {
                        "page_number": page_info["page_number"],
                        "image_size": page_info["image_size"],
                        "comprehensive_analysis": visual_analysis,
                        "visual_content_types": content_types,
                        "analysis_success": True
                    }
                    
                    results["processed_pages"].append(page_result)
                    results["success_count"] += 1
                    
                    # 추출된 시각요소들을 별도로 저장
                    results["extracted_visual_elements"].append({
                        "page": page_info["page_number"],
                        "visual_content": visual_analysis,
                        "content_types": content_types
                    })
                    
                    logger.info(f"페이지 {page_info['page_number']} 분석 완료")
                    logger.info(f"  - 수식: {content_types.get('formulas', 0)}개")
                    logger.info(f"  - 표: {content_types.get('tables', 0)}개") 
                    logger.info(f"  - 그래프: {content_types.get('graphs', 0)}개")
                    logger.info(f"  - 그림: {content_types.get('figures', 0)}개")
                    
                except Exception as e:
                    logger.error(f"페이지 {page_info['page_number']} 분석 실패: {e}")
                    
                    error_result = {
                        "page_number": page_info["page_number"],
                        "error": str(e),
                        "analysis_success": False
                    }
                    
                    results["processed_pages"].append(error_result)
                    results["failure_count"] += 1
            
            # 결과 요약
            results["visual_content_summary"] = all_content_types
            results["summary"] = {
                "total_pages_processed": pages_to_process,
                "success_rate": f"{results['success_count']}/{pages_to_process} ({results['success_count']/max(1,pages_to_process)*100:.1f}%)",
                "total_visual_elements": sum(all_content_types.values())
            }
            
            return results
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {e}")
            raise
    
    def process_supporting_pdf(self, max_pages=5) -> Dict[str, Any]:
        """
        Supporting PDF 전용 처리 함수
        """
        pdf_path = "pdf/Supporting_information_revised version.pdf"
        return self.process_pdf(pdf_path, max_pages=max_pages)

def main():
    parser = argparse.ArgumentParser(description="GPT-4o-mini로 종합적 시각요소 추출")
    parser.add_argument("--pdf", type=str, help="처리할 PDF 파일 경로")
    parser.add_argument("--output", type=str, default="gpt4_comprehensive_visual_result.json",
                       help="결과 저장 파일명")
    parser.add_argument("--supporting", action="store_true",
                       help="Supporting PDF 전용 처리")
    parser.add_argument("--max-pages", type=int, default=5,
                       help="처리할 최대 페이지 수 (비용 절약)")
    
    args = parser.parse_args()
    
    try:
        # GPT-4o-mini 종합 시각요소 처리기 초기화
        processor = GPT4ComprehensiveVisualProcessor()
        
        if args.supporting:
            # Supporting PDF 처리
            results = processor.process_supporting_pdf(args.max_pages)
        elif args.pdf:
            # 일반 PDF 처리
            results = processor.process_pdf(args.pdf, args.max_pages)
        else:
            print("--pdf 또는 --supporting 옵션을 지정해주세요")
            return
        
        # 결과 저장
        output_path = args.output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과가 {output_path}에 저장되었습니다")
        
        # 요약 출력
        print(f"\n=== GPT-4o-mini 종합 시각요소 추출 결과 ===")
        print(f"PDF 파일: {results['source_pdf']}")
        print(f"처리된 페이지: {results['summary']['total_pages_processed']}개")
        print(f"성공률: {results['summary']['success_rate']}")
        print(f"총 시각요소: {results['summary']['total_visual_elements']}개")
        
        print(f"\n=== 시각요소 유형별 통계 ===")
        for element_type, count in results["visual_content_summary"].items():
            print(f"- {element_type}: {count}개")
        
        # 성공한 페이지의 시각요소 결과 출력 (요약)
        print(f"\n=== 추출된 시각요소 미리보기 ===")
        for visual_data in results["extracted_visual_elements"][:2]:  # 처음 2페이지만
            print(f"\n--- 페이지 {visual_data['page']} ---")
            content = visual_data['visual_content']
            print(content[:600] + "..." if len(content) > 600 else content)
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
