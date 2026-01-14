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

# --- [1] 구글 드라이브 링크 변환 및 데이터 로드 로직 (기존과 동일) ---
def get_direct_url(url):
    if not url or str(url) == 'nan' or 'drive.google.com' not in url:
        return url
    if 'file/d/' in url:
        file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    else: return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    st.error(f"❌ 파일을 찾을 수 없습니다: {target_name}")
    st.stop()

@st.cache_resource
def init_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    return model, feature_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# --- [2] 매칭 보조 함수 ---
def get_digits(text):
    if not text or pd.isna(text): return ""
    return "".join(re.findall(r'\d+', str(text)))

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f_code, l_no, p_name = str(row['상품코드']).strip(), str(row['Lab No']).strip(), str(row['상품명']).strip()
        k_lab, k_formal = get_digits(l_no), get_digits(f_code)
        val = {'formal': f_code, 'name': p_name}
        if k_lab: mapping[k_lab] = val
        if k_formal: mapping[k_formal] = val
    return mapping

master_map = get_master_map()

# --- [3] UI 구성 ---
st.set_page_config(layout="wide", page_title="자재 패턴 매칭")
st.title("🏭 자재 패턴 검색 및 실시간 재고 확인")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])

if uploaded:
    # 🖼️ [추가 기능] 타겟 이미지 펼치고 닫기 (결과 분석 중에도 확인 가능)
    # 처음에는 펼쳐져 있게(expanded=True) 설정했습니다.
    with st.expander("📸 내가 업로드한 타겟 이미지 확인", expanded=True):
        col_target, col_empty = st.columns([1, 2])
        with col_target:
            st.image(uploaded, use_container_width=True, caption="검색의 기준 이미지")
        with col_empty:
            st.write("위 이미지를 기준으로 유사한 패턴을 검색합니다.")
            st.write("결과를 보실 때 이 창을 접으면 화면을 더 넓게 쓸 수 있습니다.")

    with st.spinner('유사 패턴과 실재고 대조 중...'):
        target_img = Image.open(uploaded).convert('RGB').resize((224, 224))
        x = image.img_to_array(target_img)
        x = np.expand_dims(x, axis=0)
        query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
        
        db_names, db_vecs = list(feature_db.keys()), np.array(list(feature_db.values()))
        sims = cosine_similarity(query_vec, db_vecs).flatten()
        
        results = []
        for i in range(len(db_names)):
            fname = db_names[i]
            info = master_map.get(get_digits(fname), {'formal': fname, 'name': '정보 없음'})
            formal_code = info['formal']
            qty = agg_stock.get(formal_code.strip().upper(), 0)
            url_row = df_path[df_path['파일명'] == fname]
            url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
            results.append({'formal': formal_code, 'name': info['name'], 'score': sims[i], 'stock': qty, 'url': url})
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)

    # --- [4] 결과 출력 ---
    def display_card(item, idx):
        st.markdown(f"**{idx}. {item['formal']}**")
        st.write(f"품명: {item['name']}")
        st.write(f"유사도: {item['score']:.1%}")
        with st.expander("🖼️ 이미지 보기", expanded=False):
            if item['url']:
                try:
                    res = requests.get(get_direct_url(item['url']), timeout=10)
                    st.image(Image.open(BytesIO(res.content)), use_container_width=True)
                except: st.write("❌ 이미지 로드 실패")
            else: st.write("이미지 없음")
        
        if item['stock'] >= 100: st.success(f"재고: {item['stock']:,}m")
        else: st.write(f"재고: {item['stock']:,}m")

    tab1, tab2 = st.tabs(["📊 전체 검색 결과", "✅ 재고 있음 (100m↑)"])
    with tab1:
        cols = st.columns(5)
        for i, r in enumerate(results[:10]):
            with cols[i % 5]: display_card(r, i + 1)
    with tab2:
        in_stock = [r for r in results if r['stock'] >= 100]
        if in_stock:
            cols = st.columns(5)
            for i, r in enumerate(in_stock[:10]):
                with cols[i % 5]: display_card(r, i + 1)
        else: st.warning("재고가 충분한 자재가 없습니다.")
