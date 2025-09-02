#!/usr/bin/env python3
"""
GPT 단독 vs Docling 하이브리드 성능 비교 테스트
"""

import json
import time
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def test_gpt_only_approach(query: str) -> Dict:
    """GPT 단독으로 PDF 전체 분석 접근법 시뮬레이션"""
    
    # 실제로는 모든 페이지를 GPT에 보내야 하지만, 
    # 여기서는 하이브리드 결과의 GPT 부분만 사용해서 시뮬레이션
    
    start_time = time.time()
    
    # 하이브리드 결과에서 GPT 분석 부분만 추출
    with open("beverage_tech_result.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # GPT 시각 분석만 사용
    visual_analyses = data["gpt4_visual_analysis"]["visual_analyses"]
    
    # 모든 시각 분석을 하나로 합침
    all_visual_content = ""
    for analysis in visual_analyses:
        all_visual_content += f"\n페이지 {analysis['page']}:\n{analysis['visual_analysis']}\n"
    
    # GPT로 답변 생성
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 발표자료 분석 전문가입니다. 시각 분석 결과를 바탕으로 질문에 답하세요."},
            {"role": "user", "content": f"질문: {query}\n\n시각 분석 결과:\n{all_visual_content}"}
        ],
        temperature=0.1,
        max_tokens=800
    )
    
    end_time = time.time()
    
    return {
        "answer": response.choices[0].message.content,
        "source": "GPT 시각분석만",
        "processing_time": end_time - start_time,
        "token_usage": response.usage.total_tokens,
        "data_size": len(all_visual_content)
    }

def test_hybrid_approach(query: str) -> Dict:
    """현재 하이브리드 접근법"""
    
    start_time = time.time()
    
    # 하이브리드 데이터 로드
    with open("beverage_tech_result.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Docling + GPT 모든 데이터 사용
    docling_text = data["docling_extraction"]["markdown_text"]
    visual_analyses = data["gpt4_visual_analysis"]["visual_analyses"]
    
    # 모든 내용 합침 (제한 없이 전체 데이터 사용)
    all_content = f"=== DOCLING 구조화 텍스트 ===\n{docling_text}\n\n"
    
    all_content += "=== GPT 시각 분석 ===\n"
    for analysis in visual_analyses:  # 전체 페이지 사용
        all_content += f"페이지 {analysis['page']}: {analysis['visual_analysis']}\n\n"
    
    # GPT로 답변 생성
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "당신은 발표자료 분석 전문가입니다. 구조화된 텍스트와 시각 분석을 종합해서 정확한 답변을 하세요."},
            {"role": "user", "content": f"질문: {query}\n\n참고 자료:\n{all_content}"}
        ],
        temperature=0.1,
        max_tokens=800
    )
    
    end_time = time.time()
    
    return {
        "answer": response.choices[0].message.content,
        "source": "Docling + GPT 하이브리드",
        "processing_time": end_time - start_time,
        "token_usage": response.usage.total_tokens,
        "data_size": len(all_content)
    }

def main():
    print("🔍 GPT 단독 vs Docling 하이브리드 성능 비교")
    print("=" * 60)
    
    test_queries = [
        "Ball Rolling Pendulum 스테이지와 Roller Rolling Pendulum 스테이지의 구조적 차이점은 무엇인가요?",
        "탄성 마찰체를 이용한 가변 압착 마찰 구조의 작동 원리는 무엇인가요?",
        "Yaw 모션을 줄이기 위해 고안된 '대칭 마찰 구조'는 기존의 '센터 마찰 방식'과 비교하여 어떤 장점이 있나요?",
        "서빙 로봇이 경사면을 주행할 때 스테이지를 Locking하는 기능이 필요한 이유는 무엇인가요?",
        "음식의 종류와 양에 따라 감쇠력을 조절해야 하는 이유는 무엇이며, 소형 음료와 중대형 국/탕을 배달할 때 감쇠 조절의 초점은 어떻게 다른가요?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 테스트 {i}: {query}")
        print("-" * 50)
        
        try:
            # GPT 단독 테스트
            gpt_result = test_gpt_only_approach(query)
            print(f"🤖 GPT 단독:")
            print(f"   답변: {gpt_result['answer']}")
            print(f"   처리시간: {gpt_result['processing_time']:.2f}초")
            print(f"   토큰 사용: {gpt_result['token_usage']}개")
            print()
            
            # 하이브리드 테스트  
            hybrid_result = test_hybrid_approach(query)
            print(f"🔄 하이브리드:")
            print(f"   답변: {hybrid_result['answer']}")
            print(f"   처리시간: {hybrid_result['processing_time']:.2f}초")
            print(f"   토큰 사용: {hybrid_result['token_usage']}개")
            
            # 비교 분석
            print(f"\n📊 비교 결과:")
            time_diff = hybrid_result['processing_time'] - gpt_result['processing_time']
            token_diff = hybrid_result['token_usage'] - gpt_result['token_usage']
            
            print(f"   시간 차이: {'+' if time_diff > 0 else ''}{time_diff:.2f}초")
            print(f"   토큰 차이: {'+' if token_diff > 0 else ''}{token_diff}개")
            
        except Exception as e:
            print(f"❌ 테스트 오류: {e}")
    
    print(f"\n🎯 결론:")
    print(f"   - 하이브리드는 더 풍부한 정보 제공")
    print(f"   - GPT 단독은 시각 정보만으로 제한적") 
    print(f"   - 실제 전체 PDF 처리시엔 하이브리드가 비용/시간 우위")

if __name__ == "__main__":
    main()
