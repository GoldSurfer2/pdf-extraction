#!/usr/bin/env python3
"""
개선된 하이브리드 멀티모달 RAG 평가
Docling(텍스트 보존) + GPT-4o-mini(시각 분석) + 답변 생성 LLM
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Any
from multimodal_rag import MultimodalRAG
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
client = OpenAI()

def split_docling_text(text: str) -> List[Dict[str, Any]]:
    """LangChain RecursiveCharacterTextSplitter를 사용한 의미 기반 Docling 텍스트 분할"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    docs = []
    
    # LangChain RecursiveCharacterTextSplitter 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,           # 청크 크기
        chunk_overlap=200,         # 청크간 오버랩으로 맥락 보존
        length_function=len,       # 길이 함수
        separators=[              # 분할 우선순위 (의미적 경계 우선)
            "\n## ",              # 마크다운 헤더 (최우선)
            "\n### ",             # 서브헤더
            "\n#### ",            # 더 작은 헤더
            "\n\n",               # 문단 구분
            "\n",                 # 줄바꿈
            ". ",                 # 문장 종료
            " ",                  # 단어 구분
            ""                    # 문자 단위 (최후)
        ],
        is_separator_regex=False   # 정규식 사용 안함
    )
    
    # 텍스트 분할 실행
    chunks = text_splitter.split_text(text)
    
    print(f"🔄 LangChain 의미 기반 청킹 결과: {len(chunks)}개 청크 생성")
    
    # 청크 크기 통계
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_length = min(chunk_lengths) if chunk_lengths else 0
    max_length = max(chunk_lengths) if chunk_lengths else 0
    
    print(f"📊 청크 크기 통계:")
    print(f"   - 평균: {avg_length:.0f}자")
    print(f"   - 최소: {min_length}자")
    print(f"   - 최대: {max_length}자")
    
    # 각 청크를 문서 객체로 변환
    for i, chunk in enumerate(chunks):
        docs.append({
            "page_content": chunk,
            "metadata": {
                "source": f"LangChain 청크 {i+1}",
                "chunk_id": i,
                "chunk_size": len(chunk),
                "type": "langchain_semantic_chunk"
            }
        })
    
    return docs

def generate_answer(query: str, retrieved_docs: List[str]) -> Dict[str, Any]:
    """검색된 문서를 기반으로 답변 생성 (시간 및 토큰 정보 포함)"""
    start_time = time.time()
    
    context = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs[:3])])
    
    system_message = """당신은 한국어 문서 분석 전문가입니다. 
주어진 문서들을 바탕으로 질문에 대해 정확하고 구체적인 답변을 제공하세요.

답변 규칙:
1. 문서에 명시된 구체적 정보(숫자, 날짜, 이름 등)를 정확히 인용하세요
2. 문서에 없는 정보는 추측하지 말고 "문서에 명시되지 않았습니다"라고 하세요  
3. 답변은 간결하고 핵심적으로 작성하세요
4. 가능한 한국어로 답변하세요"""

    user_message = f"""질문: {query}

참고 문서들:
{context}

위 문서들을 바탕으로 질문에 대한 정확한 답변을 제공해주세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        end_time = time.time()
        
        return {
            "answer": response.choices[0].message.content,
            "processing_time": end_time - start_time,
            "token_usage": response.usage.total_tokens,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
    except Exception as e:
        logger.error(f"답변 생성 실패: {e}")
        end_time = time.time()
        return {
            "answer": f"답변 생성 중 오류가 발생했습니다: {e}",
            "processing_time": end_time - start_time,
            "token_usage": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }

def main(input_file: str = "beverage_tech_result.json", custom_queries: List[str] = None):
    """
    하이브리드 시스템 기반 멀티모달 RAG 테스트
    
    Args:
        input_file: 하이브리드 처리 결과 JSON 파일 경로
        custom_queries: 사용자 정의 질문 리스트 (None이면 기본 질문 사용)
    """
    try:
        # 1. 하이브리드 결과 로딩
        print(f"📂 입력 파일: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            hybrid_results = json.load(f)
        
        # 2. 멀티모달 문서 생성
        multimodal_docs = []
        
        # Docling 텍스트 문서 (세밀한 섹션별 분할)
        docling_text = hybrid_results["docling_extraction"]["markdown_text"]
        docling_docs = split_docling_text(docling_text)
        multimodal_docs.extend(docling_docs)
        
        # GPT-4o-mini 페이지별 시각 분석
        gpt_analysis = hybrid_results.get("gpt4_visual_analysis", {})
        visual_analyses = gpt_analysis.get("visual_analyses", [])
        visual_count = 0
        
        for analysis_data in visual_analyses:
            page_num = analysis_data.get("page")
            visual_analysis = analysis_data.get("visual_analysis", "")
            has_visuals = analysis_data.get("has_visuals", False)
            
            if visual_analysis and visual_analysis.strip() and has_visuals:
                doc = {
                    "page_content": f"페이지 {page_num} 시각 요소 분석:\n{visual_analysis}",
                    "metadata": {
                        "source": "GPT4o-mini_visual",
                        "page": page_num,
                        "type": "visual_analysis",
                        "has_visuals": has_visuals
                    }
                }
                multimodal_docs.append(doc)
                visual_count += 1
        
        print(f"🔥 개선된 하이브리드 멀티모달 RAG 구축")
        print(f"📊 총 문서: {len(multimodal_docs)}개")
        print(f"🎯 GPT 시각 분석: {visual_count}개 페이지") 
        print(f"📝 Docling 텍스트 섹션: {len(docling_docs)}개")
        print(f"📄 원본 텍스트 길이: {len(docling_text):,}자")
        
        # 4. RAG 시스템 생성 (GPU 0번 사용 - 메모리 여유 충분)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0번 사용
        
        complete_rag = MultimodalRAG(
            text_model_name="intfloat/multilingual-e5-large",
            vision_model_name="openai/clip-vit-base-patch32",
            backend="faiss"
        )
        
        texts = [doc["page_content"] for doc in multimodal_docs]
        metadatas = [doc["metadata"] for doc in multimodal_docs]
        complete_rag.add_text_documents(texts, metadatas)
        
        # 5. 테스트 질문 설정
        if custom_queries:
            test_queries = custom_queries
            print(f"🎯 사용자 정의 질문 {len(test_queries)}개 사용")
        else:
            # 기본 발표자료 관련 테스트 질문
            test_queries = [
                "Ball Rolling Pendulum 스테이지와 Roller Rolling Pendulum 스테이지의 구조적 차이점은 무엇인가요?",
        "탄성 마찰체를 이용한 가변 압착 마찰 구조의 작동 원리는 무엇인가요?",
        "Yaw 모션을 줄이기 위해 고안된 '대칭 마찰 구조'는 기존의 '센터 마찰 방식'과 비교하여 어떤 장점이 있나요?",
        "서빙 로봇이 경사면을 주행할 때 스테이지를 Locking하는 기능이 필요한 이유는 무엇인가요?",
        "음식의 종류와 양에 따라 감쇠력을 조절해야 하는 이유는 무엇이며, 소형 음료와 중대형 국/탕을 배달할 때 감쇠 조절의 초점은 어떻게 다른가요?"
            ]
            print(f"🎯 기본 질문 {len(test_queries)}개 사용")
        
        print(f"\n🧪 개선된 하이브리드 멀티모달 RAG 테스트")
        print("=" * 60)
        
        # 전체 성능 지표 추적
        total_time = 0
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for i, query in enumerate(test_queries, 1):
            try:
                # 1. 문서 검색
                results = complete_rag.search_text(query, n_results=5)
                docs = results.get('documents', [[]])[0]
                
                print(f"\n🔍 질문 {i}: {query}")
                print("-" * 50)
                
                if docs:
                    # 2. LLM으로 답변 생성 (시간/토큰 측정)
                    result = generate_answer(query, docs)
                    answer = result['answer']
                    processing_time = result['processing_time']
                    token_usage = result['token_usage']
                    input_tokens = result['input_tokens']
                    output_tokens = result['output_tokens']
                    
                    # 전체 지표에 누적
                    total_time += processing_time
                    total_tokens += token_usage
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    
                    print(f"💡 답변: {answer}")
                    print(f"⏱️  처리 시간: {processing_time:.2f}초")
                    print(f"🪙 토큰 사용: {token_usage}개 (입력: {input_tokens}, 출력: {output_tokens})")
                    
                    # 3. 참고한 문서 정보와 출처 표시
                    metadatas = results.get('metadatas', [[]])[0]
                    distances = results.get('distances', [[]])[0]
                    
                    print(f"\n📄 참고한 상위 문서:")
                    used_docs = min(3, len(docs))  # 실제 사용된 문서만 표시
                    
                    for j in range(used_docs):
                        doc = docs[j]
                        metadata = metadatas[j] if j < len(metadatas) else {}
                        distance = distances[j] if j < len(distances) else "N/A"
                        
                        # 출처 정보 추출
                        source = metadata.get('source', '알 수 없음')
                        doc_type = metadata.get('type', '일반')
                        
                        if source == 'Docling_section':
                            section = metadata.get('section', 'N/A')
                            location = f"Docling 섹션 {section}"
                        elif source == 'GPT4o-mini_visual':
                            page = metadata.get('page', 'N/A')
                            location = f"GPT 페이지 {page} 분석"
                        else:
                            location = f"{source} ({doc_type})"
                        
                        preview = doc[:120].replace('\n', ' ').strip()
                        similarity = f"{(1-float(distance))*100:.1f}%" if distance != "N/A" else "N/A"
                        
                        print(f"   {j+1}. 📍 {location} (유사도: {similarity})")
                        print(f"      💬 {preview}...")
                        print()
                else:
                    print("💡 답변: 관련 정보를 찾을 수 없습니다.")
                    
            except Exception as e:
                print(f"❌ 질문 {i} 처리 중 오류: {e}")
        
        print("\n" + "=" * 60)
        
        print(f"\n🎉 개선된 하이브리드 멀티모달 RAG 완료!")
        print(f"✅ 세밀한 문서 분할 + 검색 + LLM 답변 생성 파이프라인 성공!")
        print(f"📊 Docling ({len(docling_docs)}개 섹션) + GPT-4o-mini ({visual_count}개 페이지) 통합")
        
        # 전체 성능 요약
        print(f"\n📈 전체 성능 요약")
        print("=" * 40)
        print(f"⏱️  총 처리 시간: {total_time:.2f}초 (평균: {total_time/len(test_queries):.2f}초)")
        print(f"🪙 총 토큰 사용: {total_tokens}개 (평균: {total_tokens/len(test_queries):.0f}개)")
        print(f"   - 입력 토큰: {total_input_tokens}개")
        print(f"   - 출력 토큰: {total_output_tokens}개")
        print(f"🎯 처리 질문 수: {len(test_queries)}개")
        print(f"💰 예상 비용: ${total_tokens * 0.000015:.4f} (GPT-4o-mini 기준)")
        
    except Exception as e:
        logger.error(f"완전한 멀티모달 RAG 구축 실패: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="개선된 하이브리드 멀티모달 RAG 평가")
    parser.add_argument("--input", type=str, default="beverage_tech_result.json", 
                        help="하이브리드 처리 결과 JSON 파일 경로")
    parser.add_argument("--queries", nargs='+', default=None,
                        help="사용자 정의 질문 리스트 (공백으로 구분)")
    
    args = parser.parse_args()
    
    main(input_file=args.input, custom_queries=args.queries)
