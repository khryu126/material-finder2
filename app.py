import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
from PIL import Image, ImageEnhance
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_cropper import st_cropper # 안정적인 크롭 라이브러리 사용

# --- [1] 기본 유틸리티 함수 ---
def get_direct_url(url):
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: return url
    if 'file/d/' in url: file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url: file_id = url.split('id=')[1].split('&')[0]
    else: return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    st.error(f"❌ {target_name} 파일을 찾을 수 없습니다.")
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

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f, l, n = str(row['상품코드']).strip(), str(row['Lab No']).strip(), str(row['상품명']).strip()
        val = {'formal': f, 'name': n}
        if get_digits(l): mapping[get_digits(l)] = val
        if get_digits(f): mapping[get_digits(f)] = val
    return mapping

master_map = get_master_map()

# --- [2] 이미지 보정 함수 ---
def apply_filters(img, lighting, surface, brightness, sharpness):
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split()
        b = b.point(lambda i: i * 1.2)
        img = Image.merge('RGB', (r, g, b))
    elif lighting == '형광등 (푸른/녹색 조명)':
        r, g, b = img.split()
        r = r.point(lambda i: i * 1.1)
        img = Image.merge('RGB', (r, g, b))
    
    enhancer_con = ImageEnhance.Contrast(img)
    if surface == '하이그로시 (반사 심함)':
        img = enhancer_con.enhance(1.5)
    elif surface == '매트/엠보 (무광)':
        img = enhancer_con.enhance(1.2)
        
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
    return img

# --- [3] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색 (Simple Mode)")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("자재 이미지를 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'])

if uploaded:
    st.markdown("### 🛠️ 이미지 전처리")
    
    # 설정 옵션은 그대로 유지
    with st.expander("📸 촬영 환경 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            source_type = st.radio("원본 종류", ['사진 촬영본', '이미지 파일 (스캔/디지털)'])
        with c2:
            lighting = st.selectbox("조명 색상", ['일반/자연광', '백열등 (누런 조명)', '형광등 (푸른/녹색 조명)'], disabled=(source_type!='사진 촬영본'))
        with c3:
            surface = st.selectbox("표면 재질", ['일반', '하이그로시 (반사 심함)', '매트/엠보 (무광)'], disabled=(source_type!='사진 촬영본'))
        with c4:
            rotation = st.radio("회전", [0, 90, 180, 270], horizontal=True, format_func=lambda x: f"↩️ {x}도" if x else "원본")

        c5, c6 = st.columns(2)
        with c5:
            brightness = st.slider("💡 밝기", 0.5, 2.0, 1.0, 0.1) if source_type == '사진 촬영본' else 1.0
        with c6:
            sharpness = st.slider("🔪 선명도", 0.0, 3.0, 1.5, 0.1) if source_type == '사진 촬영본' else 1.0

    # 이미지 로드 및 회전
    try:
        original_image = Image.open(uploaded).convert('RGB')
        if rotation != 0:
            original_image = original_image.rotate(-rotation, expand=True)
    except:
        st.error("이미지를 불러올 수 없습니다.")
        st.stop()

    # 4점 투영 대신 -> [사각형 크롭] 적용
    st.info("👇 **패턴이 잘 보이는 영역을 드래그해서 선택하세요.** (배경 제거)")
    
    # 미리 보정된 이미지를 크로퍼에 넣습니다 (눈으로 확인하기 편하게)
    preview_img = apply_filters(original_image.copy(), lighting, surface, brightness, sharpness)
    
    # st_cropper는 매우 안정적입니다.
    cropped_img = st_cropper(
        preview_img,
        realtime_update=True,
        box_color='#FF0000',
        aspect_ratio=None
    )
    
    st.caption("선택된 영역이 바로 분석됩니다.")

    if st.button("🔍 검색 시작", type="primary"):
        with st.spinner('AI 분석 중...'):
            # 이미 필터가 적용된 이미지를 크롭했으므로 바로 분석
            x = image.img_to_array(cropped_img.resize((224, 224)))
            x = np.expand_dims(x, axis=0)
            query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
            
            db_names, db_vecs = list(feature_db.keys()), np.array(list(feature_db.values()))
            sims = cosine_similarity(query_vec, db_vecs).flatten()
            
            results = []
            for i in range(len(db_names)):
                fname = db_names[i]
                info = master_map.get(get_digits(fname), {'formal': fname, 'name': '정보 없음'})
                formal = info['formal']
                qty = agg_stock.get(formal.strip().upper(), 0)
                
                url_row = df_path[df_path['추출된_품번'].apply(get_digits) == get_digits(fname)]
                if url_row.empty: url_row = df_path[df_path['파일명'] == fname]
                url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                
                results.append({'formal': formal, 'name': info['name'], 'score': sims[i], 'stock': qty, 'url': url})
            
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            st.session_state['search_results'] = results
            st.session_state['search_done'] = True

    if st.session_state.get('search_done'):
        st.markdown("---")
        results = st.session_state['search_results']
        def display_card(item, idx):
            st.markdown(f"**{idx}. {item['formal']}**")
            st.write(f"{item['name']}")
            st.caption(f"유사도: {item['score']:.1%}")
            if item['url']:
                st.markdown(f"🔗 [**고화질 원본**]({item['url']})")
                with st.expander("🖼️ 펼치기", expanded=False):
                    try:
                        r = requests.get(get_direct_url(item['url']), timeout=5)
                        st.image(Image.open(BytesIO(r.content)), use_container_width=True)
                    except: st.write("로딩 실패")
            else: st.write("이미지 없음")
            if item['stock'] >= 100: st.success(f"{item['stock']:,}m")
            else: st.write(f"{item['stock']:,}m")

        t1, t2 = st.tabs(["📊 전체 결과", "✅ 재고 보유 (100m↑)"])
        with t1:
            cols = st.columns(5)
            for i, r in enumerate(results[:10]):
                with cols[i%5]: display_card(r, i+1)
        with t2:
            hits = [r for r in results if r['stock'] >= 100]
            if hits:
                cols = st.columns(5)
                for i, r in enumerate(hits[:10]):
                    with cols[i%5]: display_card(r, i+1)
            else: st.warning("재고 보유 자재 없음")
