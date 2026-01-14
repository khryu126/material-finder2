import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# --- [1] 구글 드라이브 직접 다운로드 링크 변환 ---
def get_direct_url(url):
    if not url or str(url) == 'nan' or 'drive.google.com' not in url:
        return url
    if 'file/d/' in url:
        file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    else:
        return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

# --- [2] CSV 로드 및 인코딩 해결 (대소문자 무시) ---
def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
                try:
                    return pd.read_csv(f, encoding=enc)
                except: continue
    st.error(f"❌ 파일을 찾을 수 없습니다: {target_name}")
    st.stop()

@st.cache_resource
def init_resources():
    # AI 모델 로드
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # 지문 피클 로드
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    
    # 데이터 로드
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 재고 데이터 전처리
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    
    # 날짜 추출
    stock_date = "확인불가"
    if '정산일자' in df_stock.columns:
        d = str(int(df_stock['정산일자'].max()))
        stock_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        
    return model, feature_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# --- [3] 매칭 로직 ---
def get_only_digits(text):
    if not text or pd.isna(text): return ""
    return "".join(re.findall(r'\d+', str(text)))

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f_code = str(row['상품코드']).strip()
        l_no = str(row['Lab No']).strip()
        p_name = str(row['상품명']).strip()
        k_lab = get_only_digits(l_no)
        k_formal = get_only_digits(f_code)
        val = {'formal': f_code, 'name': p_name}
        if k_lab: mapping[k_lab] = val
        if k_formal: mapping[k_formal] = val
    return mapping

master_map = get_master_map()

# --- [4] UI 및 검색 결과 표시 ---
st.set_page_config(layout="wide", page_title="자재 패턴 매칭")
st.title("🏭 자재 패턴 검색 및 실시간 재고 확인")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("사진을 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'])

if uploaded:
    # 사용자 이미지 특징 추출
    target_img = Image.open(uploaded).convert('RGB').resize((224, 224))
    st.image(uploaded, width=250, caption="조회 패턴")

    with st.spinner('분석 중...'):
        x = image.img_to_array(target_img)
        x = np.expand_dims(x, axis=0)
        query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
        
        db_names = list(feature_db.keys())
        db_vecs = np.array(list(feature_db.values()))
        sims = cosine_similarity(query_vec, db_vecs).flatten()
        
        results = []
        for i in range(len(db_names)):
            fname = db_names[i]
            core = get_only_digits(fname)
            info = master_map.get(core, {'formal': fname, 'name': '정보 없음'})
            
            # 정밀 재고 매칭 (정식 품번 기준)
            formal_code = info['formal']
            stock_key = formal_code.strip().upper()
            qty = agg_stock.get(stock_key, 0)
            
            url_row = df_path[df_path['파일명'] == fname]
            url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
            
            results.append({
                'formal': formal_code, 'name': info['name'],
                'score': sims[i], 'stock': qty, 'url': url
            })
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)

    # --- [이미지 출력 함수] ---
    def display_card(item, idx):
        st.markdown(f"**{idx}. {item['formal']}**")
        st.write(f"품명: {item['name']}")
        st.write(f"유사도: {item['score']:.1%}")
        
        # 🖼️ 이미지 미리보기 펼치기/접기 (대리님이 요청하신 방식)
        with st.expander("🖼️ 이미지 보기", expanded=False):
            if item['url']:
                try:
                    # [수정포인트] 기존 스펙체크 방식처럼 requests로 이미지를 직접 가져옵니다.
                    direct_url = get_direct_url(item['url'])
                    response = requests.get(direct_url, timeout=15)
                    img_data = Image.open(BytesIO(response.content))
                    st.image(img_data, use_container_width=True)
                    st.caption(f"🔗 [원본 링크]({item['url']})")
                except:
                    st.write("❌ 이미지를 표시할 수 없습니다.")
            else:
                st.write("이미지 정보 없음")
        
        if item['stock'] >= 100:
            st.success(f"재고: {item['stock']:,}m")
        else:
            st.write(f"재고: {item['stock']:,}m")

    tab1, tab2 = st.tabs(["📊 전체 검색 결과", "✅ 재고 보유 (100m↑)"])
    
    with tab1:
        cols = st.columns(5)
        for i, r in enumerate(results[:10]):
            with cols[i % 5]:
                display_card(r, i + 1)

    with tab2:
        stock_hits = [r for r in results if r['stock'] >= 100]
        if stock_hits:
            cols = st.columns(5)
            for i, r in enumerate(stock_hits[:10]):
                with cols[i % 5]:
                    display_card(r, i + 1)
        else:
            st.warning("재고가 100m 이상인 자재가 없습니다.")
