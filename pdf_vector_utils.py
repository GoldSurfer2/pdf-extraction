"""
PDF 벡터 정보 활용 유틸리티
pdfplumber를 사용한 선분/사각형/글리프 추출
"""
import pdfplumber
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PDFVectorExtractor:
    """PDF 벡터 정보 추출기"""
    
    def __init__(self):
        self.angle_tolerance = 5  # 도 단위
    
    def extract_page_vectors(self, pdf_path: str, page_idx: int) -> Dict:
        """페이지별 벡터 정보 추출"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_idx >= len(pdf.pages):
                    return self._empty_vectors()
                
                page = pdf.pages[page_idx]
                
                return {
                    'lines': self._extract_lines(page),
                    'rects': self._extract_rects(page),
                    'chars': self._extract_chars(page),
                    'page_size': (page.width, page.height)
                }
                
        except Exception as e:
            logger.warning(f"벡터 추출 실패 (page {page_idx}): {e}")
            return self._empty_vectors()
    
    def _extract_lines(self, page) -> List[Dict]:
        """선분 추출 및 분류"""
        lines = []
        
        for line in page.lines:
            try:
                # 시작점과 끝점
                x0, y0 = line['x0'], line['y0']
                x1, y1 = line['x1'], line['y1']
                
                # 각도 계산
                angle = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
                
                # 방향 분류
                if abs(angle) <= self.angle_tolerance or abs(angle - 180) <= self.angle_tolerance:
                    orientation = 'horizontal'
                elif abs(angle - 90) <= self.angle_tolerance:
                    orientation = 'vertical'
                else:
                    orientation = 'diagonal'
                
                # 길이 계산
                length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                
                lines.append({
                    'bbox': [x0, y0, x1, y1],
                    'orientation': orientation,
                    'angle': angle,
                    'length': length,
                    'thickness': line.get('linewidth', 1)
                })
                
            except Exception:
                continue
        
        return lines
    
    def _extract_rects(self, page) -> List[Dict]:
        """사각형 추출"""
        rects = []
        
        for rect in page.rects:
            try:
                rects.append({
                    'bbox': [rect['x0'], rect['y0'], rect['x1'], rect['y1']],
                    'width': rect['x1'] - rect['x0'],
                    'height': rect['y1'] - rect['y0'],
                    'area': (rect['x1'] - rect['x0']) * (rect['y1'] - rect['y0'])
                })
            except Exception:
                continue
        
        return rects
    
    def _extract_chars(self, page) -> List[Dict]:
        """문자 글리프 추출"""
        chars = []
        
        for char in page.chars:
            try:
                chars.append({
                    'bbox': [char['x0'], char['y0'], char['x1'], char['y1']],
                    'text': char['text'],
                    'fontname': char.get('fontname', ''),
                    'size': char.get('size', 0)
                })
            except Exception:
                continue
        
        return chars
    
    def _empty_vectors(self) -> Dict:
        """빈 벡터 정보"""
        return {
            'lines': [],
            'rects': [],
            'chars': [],
            'page_size': (0, 0)
        }
    
    def find_table_grid(self, vectors: Dict, region_bbox: List[float]) -> Dict:
        """영역 내 표 격자 구조 분석"""
        try:
            x0, y0, x1, y1 = region_bbox
            
            # 영역 내 선분 필터링
            h_lines = []
            v_lines = []
            
            for line in vectors['lines']:
                lx0, ly0, lx1, ly1 = line['bbox']
                
                # 선분이 영역과 교차하는지 확인
                if self._line_intersects_region(line['bbox'], region_bbox):
                    if line['orientation'] == 'horizontal':
                        h_lines.append(line)
                    elif line['orientation'] == 'vertical':
                        v_lines.append(line)
            
            # 교차점 계산
            intersections = self._find_intersections(h_lines, v_lines)
            
            # 격자 규칙성 분석
            grid_regularity = self._analyze_grid_regularity(h_lines, v_lines, region_bbox)
            
            return {
                'horizontal_lines': len(h_lines),
                'vertical_lines': len(v_lines),
                'intersections': len(intersections),
                'grid_regularity': grid_regularity,
                'is_table_like': len(intersections) >= 4 and grid_regularity > 0.5
            }
            
        except Exception as e:
            logger.warning(f"격자 분석 실패: {e}")
            return {
                'horizontal_lines': 0,
                'vertical_lines': 0,
                'intersections': 0,
                'grid_regularity': 0,
                'is_table_like': False
            }
    
    def _line_intersects_region(self, line_bbox: List[float], region_bbox: List[float]) -> bool:
        """선분이 영역과 교차하는지 확인"""
        lx0, ly0, lx1, ly1 = line_bbox
        rx0, ry0, rx1, ry1 = region_bbox
        
        # 간단한 bbox 교차 검사
        return not (lx1 < rx0 or lx0 > rx1 or ly1 < ry0 or ly0 > ry1)
    
    def _find_intersections(self, h_lines: List[Dict], v_lines: List[Dict]) -> List[Tuple[float, float]]:
        """수평선과 수직선의 교차점 찾기"""
        intersections = []
        tolerance = 2  # 픽셀 허용 오차
        
        for h_line in h_lines:
            hx0, hy0, hx1, hy1 = h_line['bbox']
            
            for v_line in v_lines:
                vx0, vy0, vx1, vy1 = v_line['bbox']
                
                # 교차점 계산
                # 수평선: y = hy0 (hy0 ≈ hy1)
                # 수직선: x = vx0 (vx0 ≈ vx1)
                
                h_y = (hy0 + hy1) / 2
                v_x = (vx0 + vx1) / 2
                
                # 교차 조건 확인
                if (min(hx0, hx1) <= v_x <= max(hx0, hx1) and
                    min(vy0, vy1) <= h_y <= max(vy0, vy1)):
                    intersections.append((v_x, h_y))
        
        return intersections
    
    def _analyze_grid_regularity(self, h_lines: List[Dict], v_lines: List[Dict], 
                                region_bbox: List[float]) -> float:
        """격자 규칙성 분석 (0~1)"""
        try:
            if len(h_lines) < 2 or len(v_lines) < 2:
                return 0
            
            # 수평선 Y좌표 간격의 규칙성
            h_positions = sorted([(line['bbox'][1] + line['bbox'][3]) / 2 for line in h_lines])
            h_intervals = [h_positions[i+1] - h_positions[i] for i in range(len(h_positions)-1)]
            
            # 수직선 X좌표 간격의 규칙성
            v_positions = sorted([(line['bbox'][0] + line['bbox'][2]) / 2 for line in v_lines])
            v_intervals = [v_positions[i+1] - v_positions[i] for i in range(len(v_positions)-1)]
            
            # 간격의 변동계수 계산 (낮을수록 규칙적)
            h_regularity = 1 - (np.std(h_intervals) / np.mean(h_intervals)) if h_intervals else 0
            v_regularity = 1 - (np.std(v_intervals) / np.mean(v_intervals)) if v_intervals else 0
            
            # 전체 규칙성 점수
            regularity = (h_regularity + v_regularity) / 2
            return np.clip(regularity, 0, 1)
            
        except Exception:
            return 0
    
    def snap_bbox_to_grid(self, bbox: List[float], vectors: Dict, tolerance: float = 5) -> List[float]:
        """bbox를 벡터 격자에 스냅"""
        try:
            x0, y0, x1, y1 = bbox
            
            # 근처 선분 찾기
            nearby_lines = []
            for line in vectors['lines']:
                lx0, ly0, lx1, ly1 = line['bbox']
                
                # 거리 계산 (간단한 방법)
                if line['orientation'] == 'vertical':
                    line_x = (lx0 + lx1) / 2
                    if abs(line_x - x0) < tolerance:
                        x0 = line_x
                    if abs(line_x - x1) < tolerance:
                        x1 = line_x
                
                elif line['orientation'] == 'horizontal':
                    line_y = (ly0 + ly1) / 2
                    if abs(line_y - y0) < tolerance:
                        y0 = line_y
                    if abs(line_y - y1) < tolerance:
                        y1 = line_y
            
            return [x0, y0, x1, y1]
            
        except Exception:
            return bbox
