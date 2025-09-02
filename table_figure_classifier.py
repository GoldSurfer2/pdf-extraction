"""
표/그림 구분을 위한 증거 기반 분류기
말씀해주신 방법론을 바탕으로 구현
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TableFigureClassifier:
    """증거 기반 표/그림 분류기"""
    
    def __init__(self):
        # 분류 가중치 (튜닝 가능)
        self.weights = {
            'cross_pts': 1.0,
            'orth_line_density': 0.8, 
            'gridy_text_score': 0.6,
            'caption_is_table': 1.2,
            'rich_texture_score': -0.7  # 음수: 질감이 복잡하면 그림
        }
        self.threshold = 0.8  # 더 관대하게 (1.2 → 0.8)
    
    def extract_features(self, region_bbox: List[float], page_image: np.ndarray, 
                        pdf_lines: List = None, text_blocks: List = None) -> Dict[str, float]:
        """영역별 증거 특징 추출"""
        try:
            x0, y0, x1, y1 = region_bbox
            h, w = page_image.shape[:2]
            
            # 픽셀 좌표로 변환
            px0, py0 = int(x0 * w), int(y0 * h)
            px1, py1 = int(x1 * w), int(y1 * h)
            
            # ROI 추출
            roi = page_image[py0:py1, px0:px1]
            if roi.size == 0:
                return self._default_features()
            
            # 1. 직교 라인 밀도 (수평/수직선 교차점)
            cross_pts, orth_line_density = self._analyze_lines(roi)
            
            # 2. 텍스트 그리드성 (정렬된 텍스트 블록)
            gridy_text_score = self._analyze_text_grid(text_blocks, region_bbox) if text_blocks else 0
            
            # 3. 캡션 패턴 ("표" vs "그림")
            caption_is_table = self._analyze_caption_pattern(text_blocks, region_bbox) if text_blocks else 0
            
            # 4. 질감 복잡도 (에지 방향 다양성)
            rich_texture_score = self._analyze_texture(roi)
            
            return {
                'cross_pts': self._normalize(cross_pts, 0, 50),
                'orth_line_density': self._normalize(orth_line_density, 0, 0.1),
                'gridy_text_score': self._normalize(gridy_text_score, 0, 20),
                'caption_is_table': caption_is_table,
                'rich_texture_score': self._normalize(rich_texture_score, 0, 1)
            }
            
        except Exception as e:
            logger.warning(f"특징 추출 실패: {e}")
            return self._default_features()
    
    def _analyze_lines(self, roi: np.ndarray) -> Tuple[float, float]:
        """직교 라인 분석 (교차점 수, 라인 밀도)"""
        try:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = roi
            
            h, w = gray.shape
            
            # 이진화
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 수평/수직 커널
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//30), 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h//30)))
            
            # 선 추출
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
            
            # 교차점 계산
            intersection = cv2.bitwise_and(h_lines, v_lines)
            cross_pts = cv2.countNonZero(intersection)
            
            # 라인 밀도
            total_line_pixels = cv2.countNonZero(h_lines) + cv2.countNonZero(v_lines)
            orth_line_density = total_line_pixels / (w * h) if w * h > 0 else 0
            
            return cross_pts, orth_line_density
            
        except Exception:
            return 0, 0
    
    def _analyze_text_grid(self, text_blocks: List, region_bbox: List[float]) -> float:
        """텍스트 그리드성 분석 (정렬된 행/열 구조)"""
        if not text_blocks:
            return 0
        
        try:
            # 영역 내 텍스트 블록 필터링
            region_texts = []
            x0, y0, x1, y1 = region_bbox
            
            for block in text_blocks:
                if hasattr(block, 'bbox'):
                    bx0, by0, bx1, by1 = block.bbox
                    # 겹치는 영역이 있는지 확인
                    if not (bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1):
                        region_texts.append(block)
            
            if len(region_texts) < 4:  # 최소 4개 텍스트 블록 필요
                return 0
            
            # X, Y 좌표 클러스터링
            x_centers = [((b.bbox[0] + b.bbox[2]) / 2) for b in region_texts]
            y_centers = [((b.bbox[1] + b.bbox[3]) / 2) for b in region_texts]
            
            # 클러스터 개수 (열/행 개수 추정)
            x_clusters = self._cluster_1d(x_centers, tolerance=0.02)  # 2% 허용
            y_clusters = self._cluster_1d(y_centers, tolerance=0.01)  # 1% 허용
            
            # 그리드 점수: 열×행 구조가 명확할수록 높음
            grid_score = min(len(x_clusters), 8) + min(len(y_clusters), 15)
            
            return grid_score
            
        except Exception:
            return 0
    
    def _cluster_1d(self, values: List[float], tolerance: float) -> List[List[float]]:
        """1차원 좌표 클러스터링"""
        if not values:
            return []
        
        sorted_vals = sorted(values)
        clusters = []
        current_cluster = [sorted_vals[0]]
        
        for val in sorted_vals[1:]:
            if abs(val - current_cluster[-1]) <= tolerance:
                current_cluster.append(val)
            else:
                clusters.append(current_cluster)
                current_cluster = [val]
        
        clusters.append(current_cluster)
        return clusters
    
    def _analyze_caption_pattern(self, text_blocks: List, region_bbox: List[float]) -> float:
        """캡션 패턴 분석 ("표" vs "그림" 키워드)"""
        if not text_blocks:
            return 0
        
        try:
            x0, y0, x1, y1 = region_bbox
            search_margin = 0.05  # 5% 마진
            
            # 영역 위아래에서 캡션 찾기
            caption_texts = []
            for block in text_blocks:
                if hasattr(block, 'bbox') and hasattr(block, 'content'):
                    bx0, by0, bx1, by1 = block.bbox
                    
                    # 수평 겹침 + 수직 근접성 확인
                    h_overlap = not (bx1 < x0 - search_margin or bx0 > x1 + search_margin)
                    v_near = (by1 < y0 and by1 > y0 - search_margin) or (by0 > y1 and by0 < y1 + search_margin)
                    
                    if h_overlap and v_near:
                        caption_texts.append(block.content.lower())
            
            # 패턴 매칭
            table_patterns = ['표', 'table', 'tab.', '표 ', 'table ']
            figure_patterns = ['그림', 'figure', 'fig.', '도', '그림 ', 'figure ']
            
            table_score = sum(1 for text in caption_texts for pattern in table_patterns if pattern in text)
            figure_score = sum(1 for text in caption_texts for pattern in figure_patterns if pattern in text)
            
            if table_score > figure_score:
                return 1.0
            elif figure_score > table_score:
                return 0.0
            else:
                return 0.5  # 애매한 경우
                
        except Exception:
            return 0.5
    
    def _analyze_texture(self, roi: np.ndarray) -> float:
        """질감 복잡도 분석 (에지 방향 다양성)"""
        try:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = roi
            
            # Sobel 에지 검출
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 에지 방향 계산
            angles = np.arctan2(grad_y, grad_x)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 강한 에지만 고려
            strong_edges = magnitude > np.percentile(magnitude, 75)
            if np.sum(strong_edges) == 0:
                return 0
            
            # 방향 히스토그램 (8방향)
            angle_bins = np.digitize(angles[strong_edges], np.linspace(-np.pi, np.pi, 9))
            hist, _ = np.histogram(angle_bins, bins=8)
            
            # 엔트로피 계산 (방향 다양성)
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            return entropy / np.log(8)  # 정규화
            
        except Exception:
            return 0
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """값 정규화 (0~1)"""
        if max_val <= min_val:
            return 0
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def _default_features(self) -> Dict[str, float]:
        """기본 특징값"""
        return {
            'cross_pts': 0,
            'orth_line_density': 0,
            'gridy_text_score': 0,
            'caption_is_table': 0.5,
            'rich_texture_score': 0.5
        }
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """증거 기반 분류 (table vs figure)"""
        score = sum(self.weights[key] * features[key] for key in self.weights)
        
        if score >= self.threshold:
            return "table", score
        else:
            return "figure", score
    
    def classify_region(self, region_bbox: List[float], page_image: np.ndarray,
                       pdf_lines: List = None, text_blocks: List = None) -> Tuple[str, float, Dict]:
        """영역 분류 (원스톱)"""
        # 1단계: 텍스트 부재 시 강력한 거부
        if not self._has_meaningful_text(region_bbox, page_image):
            return "figure", 0.1, {"reason": "no_text_detected"}
        
        features = self.extract_features(region_bbox, page_image, pdf_lines, text_blocks)
        classification, score = self.classify(features)
        
        return classification, score, features
    
    def _has_meaningful_text(self, region_bbox: List[float], page_image: np.ndarray) -> bool:
        """영역에 의미있는 텍스트가 있는지 OCR로 직접 확인"""
        try:
            import pytesseract
            
            x0, y0, x1, y1 = region_bbox
            h, w = page_image.shape[:2]
            
            # 픽셀 좌표로 변환
            px0, py0 = int(x0 * w), int(y0 * h)
            px1, py1 = int(x1 * w), int(y1 * h)
            
            # ROI 추출
            roi = page_image[py0:py1, px0:px1]
            if roi.size == 0:
                return False
            
            # OCR 실행
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                gray_roi = roi
            
            # 이진화
            _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR로 텍스트 추출
            ocr_text = pytesseract.image_to_string(
                binary_roi, 
                lang='kor+eng',
                config='--psm 6'
            ).strip()
            
            # 텍스트 검증
            if len(ocr_text) < 3:  # 3글자 미만은 의미없음
                return False
            
            # 알파벳/한글/숫자가 포함되어야 함
            has_meaningful_chars = any(c.isalnum() or ord(c) > 127 for c in ocr_text)
            
            return has_meaningful_chars
            
        except Exception as e:
            # OCR 실패 시 보수적으로 False (텍스트 없음으로 간주)
            return False
