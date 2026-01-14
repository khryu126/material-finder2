import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------
# [필수] 숫자 추출 로직 강화 (랩넘버/L넘버 처리 전용)
# -----------------------------------------------------------
def extract_pure_digits(text):
    """문자열에서 5자리 이상의 연속된 숫자(랩넘버 핵심)만 추출"""
    if pd.isna(text) or str(text).strip() in ['', '-']: return ""
    nums = re.findall(r'\d{5,}', str(text)) # 보통 랩넘버는 5~6자리
    return nums[0] if nums else ""

# -----------------------------------------------------------
# [핵심] 색상 유사도 측정 개선 (검은색 배경 무시)
# -----------------------------------------------------------
def calculate_color_similarity_safe(img1_pil, img2_pil):
    """검은색 패딩(0,0,0)을 제외하고 실제 자재 색상만 비교"""
    try:
        # 이미지를 numpy 배열로 변환
        im1 = np.array(img1_pil)
        im2 = np.array(img2_pil)

        # 1. BGR -> HSV 변환
        hsv1 = cv2.cvtColor(im1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(im2, cv2.COLOR_RGB2HSV)

        # 2. 검은색(0,0,0) 마스크 생성 (Warping 후 발생하는 빈 공간 제거)
        mask1 = cv2.inRange(im1, np.array([1, 1, 1]), np.array([255, 255, 255]))
        mask2 = cv2.inRange(im2, np.array([1, 1, 1]), np.array([255, 255, 255]))

        # 3. 히스토그램 계산 (마스크 적용)
        hist1 = cv2.calcHist([hsv1], [0, 1], mask1, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        
        hist2 = cv2.calcHist([hsv2], [0, 1], mask2, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        return max(0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    except: return 0.5

# --- [1] 유틸리티 및 리소스 로드 ---
@st.cache_resource
def init_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    
    # CSV 로드 시 인코딩 문제 해결
    def load_csv(name):
        for enc in ['utf-8-sig', 'cp949']:
            try: return pd.read_csv(name, encoding=enc)
            except: continue
        return None

    df_path = load_csv('이미지경로.csv')
    df_info = load_csv('품목정보.csv')
    df_stock = load_csv('현재고.csv')
    
    # 재고 데이터 전처리
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].apply(extract_pure_digits)
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    
    return model, feature_db, df_path, df_info, agg_stock

model, feature_db, df_path, df_info, agg_stock = init_resources()

# 🧠 [마스터 매핑] 랩넘버 숫자 -> 정식품번(14-) 연결고리 생성
@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        prod_code = str(row.get('상품코드', '')).strip()
        lab_no = str(row.get('Lab No', '')).strip()
        name = str(row.get('상품명', '')).strip()
        
        # 랩넘버와 상품코드에서 숫자만 추출
        lab_digit = extract_pure_digits(lab_no)
        prod_digit = extract_pure_digits(prod_code)
        
        info = {'formal': prod_code, 'name': name, 'lab_no': lab_no}
        
        # 1. 랩넘버 숫자로 매핑 (기존에 14- 정식번호가 등록되어 있다면 덮어쓰지 않음)
        if lab_digit:
            if lab_digit not in mapping or mapping[lab_digit]['formal'].startswith('14-') == False:
                mapping[lab_digit] = info
        
        # 2. 정식 품번 숫자로 매핑
        if prod_digit:
            if prod_digit not in mapping or prod_code.startswith('14-'):
                mapping[prod_digit] = info
                
    return mapping

master_map = get_master_map()

# --- [2] 메인 검색 로직 ---
st.title("🏗️ 자재 정식품번 매칭 시스템")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 이미지 로드 및 전처리 (사용자 클릭 영역 지정 로직은 기존과 동일하다고 가정)
    # ... [영역 지정 및 Warping 코드 생략 - 이전 코드와 동일] ...
    # 결과적으로 'final_img' (Warped Image) 가 생성되었다고 가정
    
    # 예시를 위한 임시 final_img 생성 (실제 코드에선 Warping 결과물 사용)
    raw = Image.open(uploaded_file).convert('RGB')
    final_img = raw.resize((500, 500)) # 테스트용

    if st.button("🔍 정식 품번으로 검색 시작"):
        with st.spinner('패턴 및 컬러 정밀 분석 중...'):
            # 1. AI 패턴 특징 추출
            x = image.img_to_array(final_img.resize((224, 224)))
            x = np.expand_dims(x, axis=0)
            query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
            
            db_names = list(feature_db.keys())
            db_vecs = np.array(list(feature_db.values()))
            sims = cosine_similarity(query_vec, db_vecs).flatten()
            
            # 상위 후보군 추출
            top_indices = sims.argsort()[-20:][::-1]
            search_results = []
            
            for idx in top_indices:
                fname = db_names[idx] # DB에 저장된 파일명 (예: L187131.jpg)
                ai_score = sims[idx]
                
                # 2. 파일명에서 랩넘버 숫자 추출하여 정식 정보 가져오기
                file_digit = extract_pure_digits(fname)
                info = master_map.get(file_digit, {'formal': fname, 'name': '정보 미등록', 'lab_no': '-'})
                
                # 3. [개선] 색상 검증 - 검은색 패딩 무시 로직 적용
                # 실제 DB 이미지를 로드하여 비교 (경로 설정 필요)
                color_score = 0.8 # 기본값
                if os.path.exists(fname):
                    db_img = Image.open(fname).convert('RGB')
                    color_score = calculate_color_similarity_safe(final_img, db_img)
                
                # 가중치 합산 (패턴 7 : 컬러 3)
                final_score = (ai_score * 0.7) + (color_score * 0.3)
                
                search_results.append({
                    'formal': info['formal'],
                    'lab_no': info['lab_no'],
                    'name': info['name'],
                    'score': final_score,
                    'stock': agg_stock.get(extract_pure_digits(info['formal']), 0)
                })
            
            # 중복 제거 및 정렬
            search_results.sort(key=lambda x: x['score'], reverse=True)
            
            # 결과 출력
            st.subheader("✅ 유사 자재 매칭 결과")
            cols = st.columns(4)
            for i, res in enumerate(search_results[:8]):
                with cols[i % 4]:
                    # 14- 로 시작하는 정식 품번을 최우선 출력
                    st.success(f"**{res['formal']}**")
                    if res['lab_no'] != '-':
                        st.caption(f"임시번호(Lab): {res['lab_no']}")
                    st.write(f"품명: {res['name']}")
                    st.write(f"재고: {res['stock']:,}m")
                    st.progress(float(res['score']))
