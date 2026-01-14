import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import re
import pickle

# --- 1. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data():
    # CSV 로드
    df = pd.read_csv('품목정보.csv')
    
    # Lab No 열에서 숫자만 추출하는 함수 (예: L187131/10 -> 187131)
    def extract_num(val):
        if pd.isna(val): return ""
        match = re.search(r'(\d{5,})', str(val)) # 5자리 이상의 숫자 추출
        return match.group(1) if match else ""

    # 조회를 빠르게 하기 위해 숫자 전용 열 생성
    df['Lab_Numeric'] = df['Lab No'].apply(extract_num)
    return df

def get_formal_info(target_filename, df):
    """파일명에서 숫자를 뽑아 CSV에서 정식 품명과 품번을 찾아줌"""
    # 1. 파일명에서 숫자 추출 (예: 54130-L187131 -> 187131)
    match = re.search(r'(\d{5,})', target_filename)
    if not match:
        return target_filename, "정보 없음"
    
    target_id = match.group(1)
    
    # 2. CSV에서 해당 숫자 ID와 매칭되는 데이터 필터링
    matches = df[df['Lab_Numeric'] == target_id]
    
    if matches.empty:
        return f"Lab_{target_id}", "CSV 내 정보 없음"

    # 3. 매칭된 데이터 중 정식 품번(14-로 시작)이 있는 행을 우선 선택
    formal_row = matches[matches['상품코드'].str.startswith('14-', na=False)]
    
    if not formal_row.empty:
        row = formal_row.iloc[0]
    else:
        row = matches.iloc[0] # 없으면 첫 번째 검색 결과 사용

    return row['상품코드'], row['상품명']

# --- 2. 사이드바 및 설정 ---
st.set_page_config(page_title="자재 이미지 검색 시스템", layout="wide")
st.title("🏗️ 자재 유사 이미지 검색 (Lab No 매칭 적용)")

df_info = load_data()

# --- 3. 이미지 업로드 및 분석 ---
uploaded_file = st.file_uploader("검색할 대리석 이미지를 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 업로드 이미지 표시
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("검색 이미지")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8)
        query_img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- 4. 유사 이미지 검색 (기존 피클/지문 로직 적용 부분) ---
    # ※ 이 부분은 사용자님의 기존 지문 비교 함수(get_similar_results)를 넣으시면 됩니다.
    st.subheader("🔍 검색 결과 (유사도 높은 순)")
    
    # 예시 결과 데이터 (실제로는 지문 비교 함수에서 파일명 리스트가 넘어옴)
    # 예: results = [("187131.jpg", 0.95), ("158262.jpg", 0.88)]
    results = [("14-54130-119.jpg", 0.95), ("L187131.jpg", 0.92), ("158262.jpg", 0.88)] 

    cols = st.columns(3)
    for i, (res_filename, score) in enumerate(results):
        with cols[i % 3]:
            # 핵심: 파일명에서 정식 번호 찾아오기
            formal_code, product_name = get_formal_info(res_filename, df_info)
            
            # 결과 출력
            st.image("path_to_images/" + res_filename, use_column_width=True) # 실제 경로에 맞게 수정
            st.success(f"**순위: {i+1}**")
            st.write(f"**품번:** {formal_code}")
            st.write(f"**품명:** {product_name}")
            st.write(f"**유사도:** {score:.2%}")
            st.divider()

else:
    st.info("이미지를 업로드하면 데이터베이스에서 가장 유사한 자재의 정식 정보를 찾아드립니다.")

# --- 5. 코드 수정 가이드 ---
st.sidebar.markdown("""
### 💡 수정된 포인트
1. **Lab No 매칭**: `L187131`처럼 이름이 제각각인 임시번호를 숫자 `187131`로만 인식하여 정확히 매칭합니다.
2. **정식 품번 우선**: 검색 결과에 임시번호와 정식번호가 섞여 있을 때, **14-로 시작하는 정식 코드**를 우선적으로 가져옵니다.
3. **색상 오류 해결**: 파일명의 숫자를 색상(RGB)으로 해석하지 않고 검색 키워드로만 사용합니다.
""")
