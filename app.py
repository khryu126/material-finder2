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

# --- [보조 함수: 한글 깨짐 및 대소문자 무시 로드] ---
def load_csv_smart(target_name):
    """파일 이름 대소문자 구분 없이 찾고, 한글 인코딩(UTF-8, CP949)을 자동 해결"""
    found_file = None
    for f in os.listdir('.'):
        if f.lower() == target_name.lower():
            found_file = f
            break
    
    if not found_file:
        st.error(f"❌ 파일을 찾을 수 없습니다: {target_name}")
        st.stop()
        
    for enc in ['utf-8', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(found_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    st.error(f"❌ {target_name}의 인코딩을 판별할 수 없습니다.")
    st.stop()

# --- [1] 리소스 로드 (캐싱) ---
@st.cache_resource
def init_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 재고 수량 전처리 및 품번별 합산
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

# --- [2] 매칭 및 검색 함수 ---
def get_only_digits(text):
    if not text or pd.isna(text): return ""
    return "".join(re.findall(r'\d+', str(text)))

@st.cache_data
def get_item_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f_code = str(row['상품코드']).strip()
        l_no = str(row['Lab No']).strip()
        p_name = str(row['상품명']).strip()
        
        # 랩넘버와 정식품번 숫자를 모두 키로 사용
        k_lab = get_only_digits(l_no)
        k_formal = get_only_digits(f_code)
        
        val = {'formal': f_code, 'name': p_name}
        if k_lab: mapping[k_lab] = val
        if k_formal: mapping[k_formal] = val
    return mapping

master_map = get_item_map()

# --- [3] UI 구성 ---
st.set_page_config(layout="wide", page_title="자재 패턴 매칭")
st.title("🏭 자재 패턴 검색 및 실시간 재고 확인")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])

if uploaded:
    # 사용자 이미지 특징 추출
    target_img = Image.open(uploaded).convert('RGB').resize((224, 224))
    st.image(uploaded, width=250, caption="조회 패턴")

    with st.spinner('패턴 분석 및 재고 대조 중...'):
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
            
            # 정밀 재고 매칭 (대소문자/공백 제거 후 일치 확인)
            formal_code = info['formal']
            stock_key = formal_code.strip().upper()
            qty = agg_stock.get(stock_key, 0)
            
            # 이미지 URL
            url_row = df_path[df_path['파일명'] == fname]
            url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
            
            results.append({
                'formal': formal_code, 'name': info['name'],
                'score': sims[i], 'stock': qty, 'url': url
            })
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)

    # 결과 출력
    t1, t2 = st.tabs(["📊 전체 유사 결과", "✅ 재고 있음 (100m↑)"])
    with t1:
        cols = st.columns(5)
        for i, r in enumerate(results[:10]):
            with cols[i%5]:
                st.image(r['url'] if r['url'] else "https://via.placeholder.com/150")
                st.markdown(f"**{r['formal']}**")
                st.caption(f"유사도: {r['score']:.1%}")
                st.write(f"재고: {r['stock']:,}m")
    with t2:
        in_stock = [r for r in results if r['stock'] >= 100]
        if in_stock:
            cols = st.columns(5)
            for i, r in enumerate(in_stock[:10]):
                with cols[i%5]:
                    st.image(r['url'] if r['url'] else "https://via.placeholder.com/150")
                    st.success(f"**{r['formal']}**")
                    st.write(f"품명: {r['name']}")
                    st.write(f"**실재고: {r['stock']:,}m**")
        else:
            st.warning("재고가 100m 이상인 자재가 없습니다.")
