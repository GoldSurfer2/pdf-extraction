# PDF OCR 및 구조 분석 파이프라인

## 프로젝트 개요
- PDF 파일에서 원하는 페이지만 선택하여 OCR 및 레이아웃(구조) 분석
- EasyOCR(한글/영어) 및 Google Document AI Layout Parser를 활용
- 결과는 텍스트, JSON 등 다양한 포맷으로 저장

## 환경 준비
1. **Python 가상환경 생성 및 패키지 설치**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```

2. **Poppler 설치**
   - PDF를 이미지로 변환할 때 Poppler가 필요합니다.
   - Windows용 Poppler 바이너리는 용량이 크고, OS별로 다르므로 git에 포함하지 않습니다.
   - [Poppler 다운로드](https://github.com/oschwartz10612/poppler-windows/releases/) 후, `poppler-24.08.0/` 폴더에 압축 해제
   - 환경변수 또는 코드 내 경로(`POPLER_PATH`)를 맞춰주세요.

3. **Google Document AI 사용**
   - Google Cloud Console에서 Document AI Layout Parser 프로세서 생성
   - 서비스 계정 키(JSON) 다운로드 및 환경변수 `GOOGLE_APPLICATION_CREDENTIALS`에 경로 지정
   - 프로젝트 ID, location, processor ID를 코드에 입력

## 실행 방법
1. 원하는 PDF 파일을 `pdf/` 폴더에 넣고, `PDF_PATH`를 수정
2. `SELECTED_PAGES`에 분석할 페이지 번호(1부터 시작)를 지정
3. 아래 명령어로 실행
   ```bash
   python main.py
   ```

## 결과 파일
- `result_타임스탬프/` 폴더에 OCR 및 구조 분석 결과가 저장됩니다.
- EasyOCR 결과: `ocr_result_*.txt`, `ocr_result_*.json`
- Google Document AI 결과: `google_ocr_result_*.txt`, `google_layout_result_*.json`

## 기타
- `.venv`, `poppler-24.08.0/`, 결과 폴더 등은 git에 포함하지 않습니다.
- 패키지 목록은 `requirements.txt`로 관리합니다.

---


## Detectron2 기반 레이아웃 탐지 학습

1) DocLayNet → COCO 변환 (로컬 캐시 사용):

```bash
python tools/doclaynet_to_coco.py --output coco_dataset --splits train validation --cache_dir DocLayNet
```

2) (선택) 검증 split 소량으로 스모크 테스트:

```bash
python tools/doclaynet_to_coco.py --output coco_dataset_small --splits validation --cache_dir DocLayNet --max_images 50
```

3) 학습 종속성 설치 (CUDA wheel index는 환경에 맞게 조정 가능):

```bash
pip install -r requirements.txt
```

4) Mask R-CNN R50-FPN 학습 시작:

```bash
python train_detectron2.py
```

학습 결과는 `outputs_detectron2/` 하위에 저장됩니다.

