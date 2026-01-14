import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
import time
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------
# 🚑 [필수 패치] Streamlit 호환성 및 이미지 출력 안정화
# -----------------------------------------------------------
import streamlit.elements.image as st_image

def local_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="auto", image_id=None):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

if not hasattr(st_image, 'image_to_url'):
    st_image.image_to_url = local_image_to_url

# --- [1] 유틸리티 및 리소스 ---
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

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

def is_formal_code(code):
    if not code or pd.isna(code): return False
    pattern = r'^\d+-\d+-\d+$'
    return bool(re.match(pattern, str(code).strip()))

@st.cache_resource
def init_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 🚀 [세이브포인트 복원] 재고 매칭 키 전처리 로직
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper() 
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    
    valid_keys = set(df_path['추출된_품번'].apply(get_digits).unique())
    filtered_db = {k: v for k, v in feature_db.items() if get_digits(k) in valid_keys}
    
    return model, filtered_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f = str(row['상품코드']).strip() if pd.notna(row.get('상품코드')) else ''
        l = str(row.get('Lab No', '')).strip() if pd.notna(row.get('Lab No')) else ''
        n = str(row.get('상품명', '')).strip() if pd.notna(row.get('상품명')) else ''
        current_formal = f if f else l
        info = {'formal': current_formal, 'name': n, 'lab_no': l}
        keys = set()
        for v in [f, l]:
            d = get_digits(v)
            if d: keys.add(d)
        for k in keys:
            if k not in mapping or (is_formal_code(current_formal) and not is_formal_code(mapping[k]['formal'])):
                mapping[k] = info
    return mapping

master_map = get_master_map()

# --- [2] 이미지 처리 함수 ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# --- [3] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 패턴 검색")
st.title("🏭 스마트 자재 패턴 검색")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

# 세션 상태 초기화
if 'points' not in st.session_state: st.session_state['points'] = []
if 'search_done' not in st.session_state: st.session_state['search_done'] = False
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0
if 'upload_ready' not in st.session_state: st.session_state['upload_ready'] = False
if 'img_scale' not in st.session_state: st.session_state['img_scale'] = 1.0

# 🚀 [개선] 동적 하트비트: 업로드 준비 버튼과 움직이는 스피너
if not st.session_state['upload_ready']:
    st.info("📱 모바일 연결을 활성화하기 위해 아래 버튼을 클릭하세요.")
    if st.button("🚀 업로드 준비 및 연결 시작"):
        st.session_state['upload_ready'] = True
        st.rerun()
else:
    with st.sidebar:
        with st.spinner("💓 연결 유지 중 (활동 상태)"):
            time.sleep(0.1) # 시각적인 움직임 제공

    uploaded = st.file_uploader("📸 분석할 자재 사진 선택", type=['jpg','png','jpeg'], key=f"up_{st.session_state['uploader_key']}")

    if st.sidebar.button("🔄 전체 초기화 (Reset)"):
        for k in ['points', 'search_done', 'search_results', 'upload_ready', 'proc_img']:
            if k in st.session_state: del st.session_state[k]
        st.session_state['uploader_key'] += 1
        st.rerun()

    if uploaded:
        if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
            st.session_state['points'] = []; st.session_state['search_done'] = False
            st.session_state['current_img_name'] = uploaded.name
            with st.spinner('📸 이미지 처리 중...'):
                raw = Image.open(uploaded).convert('RGB')
                raw.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                st.session_state['proc_img'] = raw
            st.rerun()

        working_img = st.session_state['proc_img']
        
        st.markdown("### 1️⃣ 환경 설정")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1: mat_type = st.selectbox("🧱 자재 종류", ['일반', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)'])
        with col_opt2: s_mode = st.radio("🔎 검색 기준", ["🎨 컬러+패턴", "🦓 패턴 중심 (흑백)"], horizontal=True)

        st.markdown("### 2️⃣ 영역 지정")
        
        # 스케일 조절 버튼
        scale_val = st.select_slider("🔍 모바일 보기 크기 (작게 해서 점을 찍으세요)", 
                                    options=[0.3, 0.5, 0.7, 1.0], value=st.session_state['img_scale'])
        st.session_state['img_scale'] = scale_val

        # 🚀 [복원] 점 지우기 버튼 및 새로고침
        c_ref, c_del, c_auto = st.columns([1, 1, 2])
        with c_ref: 
            if st.button("🔄 화면 새로고침"): st.rerun()
        with c_del:
            if st.button("❌ 점 지우기", type="secondary"):
                st.session_state['points'] = []; st.rerun()
        with c_auto:
            if st.button("⏹️ 이미지 전체 선택 (Auto)", type="primary"):
                w, h = working_img.size
                st.session_state['points'] = [(0, 0), (w, 0), (w, h), (0, h)]; st.rerun()

        # 이미지 리사이징 및 점 그리기
        w, h = working_img.size
        display_w, display_h = int(w * scale_val), int(h * scale_val)
        display_img = working_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        
        draw_img = display_img.copy(); draw = ImageDraw.Draw(draw_img)
        
        # 🚀 [복원] 번호 표시(Point 1, 2...) 로직
        for i, p in enumerate(st.session_state['points']):
            px, py = p[0] * scale_val, p[1] * scale_val
            draw.ellipse((px-8, py-8, px+8, py+8), fill='red', outline='white', width=2)
            # 번호 텍스트 추가
            draw.text((px + 12, py - 12), str(i + 1), fill='red')

        if len(st.session_state['points']) == 4:
            pts_scaled = [(p[0]*scale_val, p[1]*scale_val) for p in st.session_state['points']]
            draw.polygon([tuple(p) for p in order_points(np.array(pts_scaled))], outline='#00FF00', width=3)

        value = streamlit_image_coordinates(draw_img, key="click_pad")
        if value:
            real_x, real_y = value['x'] / scale_val, value['y'] / scale_val
            if len(st.session_state['points']) < 4:
                new_p = (real_x, real_y)
                if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                    st.session_state['points'].append(new_p); st.rerun()

        # --- [3] 검색 분석 ---
        if len(st.session_state['points']) == 4:
            if st.button("🔍 이 영역으로 검색 시작", type="primary", use_container_width=True):
                with st.spinner('유사 패턴을 분석 중입니다...'):
                    warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
                    final_img = Image.fromarray(warped)
                    if s_mode == "🦓 패턴 중심 (흑백)": final_img = final_img.convert("L").convert("RGB")
                    
                    x = image.img_to_array(final_img.resize((224, 224)))
                    query_vec = model.predict(preprocess_input(np.expand_dims(x, axis=0)), verbose=0).flatten().reshape(1, -1)
                    
                    db_names = list(feature_db.keys()); db_vecs = np.array(list(feature_db.values()))
                    sims = cosine_similarity(query_vec, db_vecs).flatten()
                    
                    all_res = []; stock_res = []
                    seen_all = set(); seen_stock = set()
                    
                    sorted_idx = np.argsort(sims)[::-1]
                    for i in sorted_idx:
                        fname = db_names[i]
                        info = master_map.get(get_digits(fname), {'formal': fname, 'name': '정보 없음'})
                        f_code = info['formal']
                        
                        # 🚀 [세이브포인트 복원] 재고 매칭 키 (Strip + Upper)
                        f_key = f_code.strip().upper()
                        qty = agg_stock.get(f_key, 0)
                        
                        url_row = df_path[df_path['추출된_품번'].apply(get_digits) == get_digits(fname)]
                        url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                        
                        if url:
                            data = {'formal': f_code, 'name': info['name'], 'score': score, 'stock': qty, 'url': url}
                            if f_code not in seen_all and len(all_res) < 15:
                                all_res.append(data); seen_all.add(f_code)
                            # 재고 100m 이상 별도 검색 풀
                            if qty >= 100 and f_code not in seen_stock and len(stock_res) < 15:
                                stock_res.append(data); seen_stock.add(f_code)
                    
                    st.session_state['search_results'] = {'all': all_res, 'stock': stock_res}
                    st.session_state['search_done'] = True; st.rerun()

        # --- [4] 결과 카드 (액박 방지 requests 로직 유지) ---
        if st.session_state.get('search_done'):
            st.markdown("---")
            res = st.session_state['search_results']
            def draw_card(item, idx):
                st.markdown(f"**{idx}. {item['formal']}**")
                st.caption(f"{item['name']} (유사도: {item['score']:.1%})")
                with st.expander("🖼️ 이미지 확인", expanded=False):
                    try:
                        r = requests.get(get_direct_url(item['url']), timeout=5)
                        st.image(Image.open(BytesIO(r.content)), use_container_width=True)
                    except: st.write("⚠️ 로드 실패")
                if item['stock'] >= 100: st.success(f"재고: {item['stock']:,}m")
                else: st.info(f"재고: {item['stock']:,}m")

            t1, t2 = st.tabs(["📊 전체 결과", "✅ 재고 보유 유사도 (100m↑)"])
            with t1:
                cols = st.columns(5)
                for i, r in enumerate(res['all']):
                    with cols[i%5]: draw_card(r, i+1)
            with t2:
                if res['stock']:
                    cols = st.columns(5)
                    for i, r in enumerate(res['stock']):
                        with cols[i%5]: draw_card(r, i+1)
                else: st.warning("⚠️ 100m 이상 재고 품목 중 유사 자재가 없습니다.")
