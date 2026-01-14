import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# --- [보조 함수: 파일명 대소문자 무시 및 자동 로드] ---
def load_csv_ignore_case(target_name):
    """현재 폴더에서 대소문자 구분 없이 해당 CSV 파일을 찾아 로드"""
    for f in os.listdir('.'):
        if f.lower() == target_name.lower():
            return pd.read_csv(f)
    st.error(f"❌ 파일을 찾을 수 없습니다: {target_name}")
    st.stop()

# --- [1] 리소스 로드 (캐싱 적용) ---
@st.cache_resource
def load_all_resources():
    # AI 모델 (특징 추출용)
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # 지문 피클 데이터
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    
    # CSV 파일 3종 로드
    df_path = load_csv_ignore_case('이미지경로.csv')
    df_info = load_csv_ignore_case('품목정보.csv')
    df_stock = load_csv_ignore_case('현재고.csv')
    
    # --- 재고 데이터 사전 합산 ---
    # 재고수량에서 콤마 제거 및 숫자 변환
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # 품번의 공백 제거 및 대문자화 (정밀 매칭 준비)
    df_stock['품번_clean'] = df_stock['품번'].astype(str).str.strip().str.upper()
    
    # 품번별로 모든 롤(Roll) 재고 합산 -> 딕셔너리 생성
    # 예: {'14-12345-100': 550.5}
    agg_stock = df_stock.groupby('품번_clean')['재고수량'].sum().to_dict()
    
    # 재고 업데이트 날짜 (정산일자 기준)
    stock_date = "확인불가"
    if '정산일자' in df_stock.columns:
        d = str(int(df_stock['정산일자'].max()))
        stock_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    return model, feature_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = load_all_resources()

# --- [2] 핵심 매칭 함수 ---
def get_digit_key(text):
    """텍스트에서 숫자만 추출하여 매칭 키 생성"""
    if not text or pd.isna(text): return ""
    return "".join(re.findall(r'\d+', str(text)))

@st.cache_data
def build_mapping_table():
    """품목정보 기반: 랩넘버/정식번호 숫자를 정식 품번 정보로 연결"""
    mapping = {}
    for _, row in df_info.iterrows():
        f_code = str(row['상품코드']).strip()
        l_no = str(row['Lab No']).strip()
        p_name = str(row['상품명']).strip()
        
        # 랩넘버 숫자와 정식번호 숫자를 모두 키로 등록
        key_lab = get_digit_key(l_no)
        key_formal = get_digit_key(f_code)
        
        val = {'formal_code': f_code, 'item_name': p_name}
        if key_lab: mapping[key_lab] = val
        if key_formal: mapping[key_formal] = val
    return mapping

master_map = build_mapping_table()

# --- [3] 메인 UI 및 로직 ---
st.set_page_config(layout="wide", page_title="자재 패턴 매칭")
st.title("🏭 자재 패턴 유사도 및 정밀 재고 확인")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("자재 사진(JPG, PNG, TIF)을 업로드하세요", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])

if uploaded:
    # 1. 업로드 이미지 전처리
    target_img = Image.open(uploaded).convert('RGB').resize((224, 224))
    st.image(uploaded, width=300, caption="조회 패턴")

    with st.spinner('유사 패턴 분석 중...'):
        # 2. 특징값 추출 (AI)
        x = image.img_to_array(target_img)
        x = np.expand_dims(x, axis=0)
        query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
        
        # 3. 유사도 계산
        db_keys = list(feature_db.keys())
        db_vecs = np.array(list(feature_db.values()))
        sims = cosine_similarity(query_vec, db_vecs).flatten()
        
        # 4. 결과 데이터 결합
        final_list = []
        for i in range(len(db_keys)):
            fname = db_keys[i]
            score = sims[i]
            
            # 파일명에서 숫자 추출하여 정식 정보 찾기
            core_key = get_digit_key(fname)
            info = master_map.get(core_key, {'formal_code': fname, 'item_name': '정보 없음'})
            
            formal_code = info['formal_code']
            
            # [핵심 로직] 정식 품번과 재고 파일의 품번을 1:1 대조 (대소문자/공백 무시)
            match_key = formal_code.strip().upper()
            qty = agg_stock.get(match_key, 0)
            
            # 구글 드라이브 URL 연결
            url_row = df_path[df_path['파일명'] == fname]
            img_url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
            
            final_list.append({
                'formal_code': formal_code,
                'item_name': info['item_name'],
                'score': score,
                'stock': qty,
                'url': img_url
            })
        
        # 유사도 높은 순 정렬
        final_list = sorted(final_list, key=lambda x: x['score'], reverse=True)

    # --- [4] 결과 출력 ---
    tab1, tab2 = st.tabs(["📊 전체 유사 패턴", "✅ 재고 보유 (100m↑)"])
    
    with tab1:
        cols = st.columns(5)
        for i, item in enumerate(final_list[:10]):
            with cols[i % 5]:
                st.image(item['url'] if item['url'] else "https://via.placeholder.com/150")
                st.markdown(f"**{item['formal_code']}**")
                st.caption(f"유사도: {item['score']:.1%}")
                st.write(f"재고: {item['stock']:,}m")

    with tab2:
        # 합산 재고가 100 이상인 것만 필터링
        in_stock = [it for it in final_list if it['stock'] >= 100]
        if in_stock:
            cols = st.columns(5)
            for i, item in enumerate(in_stock[:10]):
                with cols[i % 5]:
                    st.image(item['url'] if item['url'] else "https://via.placeholder.com/150")
                    st.success(f"**{item['formal_code']}**")
                    st.write(f"품명: {item['item_name']}")
                    st.write(f"**실재고: {item['stock']:,}m**")
        else:
            st.warning("재고가 100m 이상인 유사 자재가 없습니다.")