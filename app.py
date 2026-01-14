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

# --- [1] 구글 드라이브 직접 다운로드 링크 변환 함수 ---
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

# --- [2] CSV 로드 및 한글 인코딩 자동 해결 (대소문자 무시) ---
def load_csv_smart(target_name):
    """파일 이름 대소문자 구분 없이 찾고, 한글 인코딩을 자동 해결"""
    found_file = None
    for f in os.listdir('.'):
        if f.lower() == target_name.lower():
            found_file = f
            break
    
    if not found_file:
        st.error(f"❌ 파일을 찾을 수 없습니다: {target_name}")
        st.stop()
        
    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(found_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    st.error(f"❌ {target_name}의 글자 형식을 판별할 수 없습니다. 파일을 확인해 주세요.")
    st.stop()

# --- [3] 리소스 로드 (AI 모델, 피클, CSV) ---
@st.cache_resource
def init_resources():
    # AI 모델 로드
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # 지문 피클 데이터 로드
    # (용량 줄이기 코드로 만든 15MB 파일을 material_features.pkl로 저장했을 것으로 가정)
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    
    # 데이터 파일 로드
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 재고 데이터 전처리
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    # 품번의 공백 제거 및 대문자화 (정밀 매칭용)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    # 품번별로 여러 롤(Roll) 재고 합산
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    
    # 재고 업데이트 날짜 추출 (정산일자 기준)
    stock_date = "확인불가"
    if '정산일자' in df_stock.columns:
        d = str(int(df_stock['정산일자'].max()))
        stock_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        
    return model, feature_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# --- [4] 매칭 보조 함수 ---
def get_digits(text):
    """텍스트에서 숫자만 추출 (매칭의 핵심)"""
    if not text or pd.isna(text): return ""
    return "".join(re.findall(r'\d+', str(text)))

@st.cache_data
def get_master_map():
    """품목정보 기반: 랩넘버/정식번호 숫자를 정식 품번 정보로 연결"""
    mapping = {}
    for _, row in df_info.iterrows():
        f_code = str(row['상품코드']).strip()
        l_no = str(row['Lab No']).strip()
        p_name = str(row['상품명']).strip()
        
        # 랩넘버와 정식품번 숫자를 키로 등록 (예: L233959 -> 233959)
        k_lab = get_digits(l_no)
        k_formal = get_digits(f_code)
        
        val = {'formal': f_code, 'name': p_name}
        if k_lab: mapping[k_lab] = val
        if k_formal: mapping[k_formal] = val
    return mapping

master_map = get_master_map()

# --- [5] UI 구성 및 검색 로직 ---
st.set_page_config(layout="wide", page_title="자재 통합 매칭 시스템")
st.title("🏭 자재 패턴 검색 및 실시간 재고 확인")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("자재 사진을 업로드하세요", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])

if uploaded:
    # 📸 타겟 이미지 접기/펴기 (결과와 비교하기 위해 상단 배치)
    with st.expander("📸 내가 업로드한 타겟 이미지 확인", expanded=True):
        col_t, col_e = st.columns([1, 2])
        with col_t:
            st.image(uploaded, use_container_width=True, caption="검색 기준 패턴")
        with col_e:
            st.write("이 이미지를 기준으로 가장 유사한 자재를 검색합니다.")
            st.write("결과를 보실 때 이 창을 접으면 화면을 넓게 사용하실 수 있습니다.")

    with st.spinner('유사 패턴 분석 및 재고 대조 중...'):
        # 1. AI 지문 추출
        target_img = Image.open(uploaded).convert('RGB').resize((224, 224))
        x = image.img_to_array(target_img)
        x = np.expand_dims(x, axis=0)
        query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
        
        # 2. 유사도 계산
        db_names = list(feature_db.keys())
        db_vecs = np.array(list(feature_db.values()))
        sims = cosine_similarity(query_vec, db_vecs).flatten()
        
        # 3. 결과 데이터 결합
        results = []
        for i in range(len(db_names)):
            fname = db_names[i]
            fname_digits = get_digits(fname)
            
            # 랩넘버 여부 확인 및 정식 정보 매칭
            info = master_map.get(fname_digits, {'formal': fname, 'name': '정보 없음'})
            formal_code = info['formal']
            
            # 정밀 재고 매칭 (정식 품번 글자가 재고 파일에 정확히 있는지 확인)
            stock_key = formal_code.strip().upper()
            qty = agg_stock.get(stock_key, 0)
            
            # [이미지 매칭] 확장자(.jpg / .tif) 무시를 위해 숫자 기반으로 URL 매칭
            url_row = df_path[df_path['추출된_품번'].apply(get_digits) == fname_digits]
            if url_row.empty:
                url_row = df_path[df_path['파일명'] == fname]
            
            url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
            
            results.append({
                'formal': formal_code,
                'name': info['name'],
                'score': sims[i],
                'stock': qty,
                'url': url
            })
        
        # 유사도 순 정렬
        results = sorted(results, key=lambda x: x['score'], reverse=True)

    # --- [6] 결과 출력 함수 (카드 형태 + Expander) ---
    def display_card(item, idx):
        st.markdown(f"**{idx}. {item['formal']}**")
        st.write(f"품명: {item['name']}")
        st.write(f"유사도: {item['score']:.1%}")
        
        # 🖼️ 이미지 접기/펴기 기능 (Requests 방식으로 로딩)
        with st.expander("🖼️ 이미지 보기", expanded=False):
            if item['url']:
                try:
                    direct_url = get_direct_url(item['url'])
                    res = requests.get(direct_url, timeout=10)
                    img_data = Image.open(BytesIO(res.content))
                    st.image(img_data, use_container_width=True)
                    st.caption(f"🔗 [원본 링크]({item['url']})")
                except:
                    st.write("❌ 이미지를 불러올 수 없습니다.")
            else:
                st.write("등록된 이미지 없음")
        
        # 재고 수량 강조
        if item['stock'] >= 100:
            st.success(f"재고: {item['stock']:,}m")
        else:
            st.write(f"재고: {item['stock']:,}m")

    # 탭 구분 (전체 vs 재고 100m 이상)
    tab1, tab2 = st.tabs(["📊 전체 검색 결과", "✅ 재고 보유 (100m↑)"])
    
    with tab1:
        cols = st.columns(5)
        for i, r in enumerate(results[:10]): # 상위 10개만 출력
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
            st.warning("유사한 패턴 중 재고가 100m 이상인 자재가 없습니다.")
