# Doc-to-Patent System Architecture

## Overview
발표자료(PPT/PDF) → 추출 → 지능형 처리 → 생성 → QA의 파이프라인을 코드 수준 산출물로 연결합니다.

## Flow
```mermaid
flowchart LR
  A[입력: PPT/PDF] --> B[구성요소 수준 추출 엔진\n(Python-pptx, PyMuPDF, Camelot/Tabula)]
  B --> C[지능형 처리 파이프라인\n(OpenCV, ROI, Diagram/Chart/Flow 분석)]
  C --> D[지식 그래프/IR]
  D --> E1[도면 설명 생성]
  D --> E2[상세 설명 생성]
  D --> E3[청구항 생성]
  E1 & E2 & E3 --> F[자동 QA 코파일럿]
  F --> G[최종 명세서/보고]
```

## IR 스키마 초안
- 파일: `schemas/ir_schema.json`
- 목적: 페이지/블록/관계/근거 링크를 일원화하여 RAG·QA에 공용 입력 제공

## 최소 실행 로드맵
- [x] DocLayNet 다운로드 스크립트
- [ ] DocLayNet 통계/미리보기 도구
- [ ] IR 스키마 파일과 유효성 검사기
- [ ] 임베딩(RAG) 스타터: PatentSBERTa + Chroma
- [ ] QA 루틴 초안(antecedent, 부호-본문 매핑 점검)

## Notes
- 데이터 용량 관리: `tools/hf_cache_clean.py`로 미사용 모델/스냅샷 정리
- 결과는 `result_YYYYMMDD_HHMMSS/`로 버저닝 유지
