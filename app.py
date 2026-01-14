import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------
# 🚑 [필수 패치] Streamlit 호환성 & 흰 화면 해결
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
    
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    
    # 🚀 [추가] 이미지 URL이 실재하는 데이터만 검색 Pool로 한정
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

# --- [2] 이미지 처리 ---
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

def apply_smart_filters(img, category, lighting, brightness, sharpness):
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split(); b = b.point(lambda i: i * 1.2); img = Image.merge('RGB', (r, g, b))
    elif lighting == '형광등 (푸른/녹색 조명)':
        r, g, b = img.split(); r = r.point(lambda i: i * 1.1); img = Image.merge('RGB', (r, g, b))
    
    en_con = ImageEnhance.Contrast(img); en_shp = ImageEnhance.Sharpness(img)
    en_bri = ImageEnhance.Brightness(img); en_col = ImageEnhance.Color(img)
    
    if category == '마루/우드 (Wood)':
        img = en_shp.enhance(2.0); img = en_con.enhance(1.1)
    elif category == '하이그로시/유광 (Glossy)':
        img = en_con.enhance(1.5); img = en_shp.enhance(1.2)
    elif category == '벽지/패브릭 (Texture)':
        img = en_shp.enhance(1.5); img = en_bri.enhance(1.1)
    elif category == '석재/콘크리트 (Stone)':
        img = en_col.enhance(0.8); img = en_shp.enhance(1.5)
    
    if brightness != 1.0: img = en_bri.enhance(brightness)
    if sharpness != 1.0: img = en_shp.enhance(sharpness)
    return img

# --- [3] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

# 세션 초기화
for key in ['points', 'search_done', 'uploader_key', 'search_results']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['points', 'search_results'] else (0 if key == 'uploader_key' else False)

with st.expander("📘 사용 가이드", expanded=False):
    st.markdown("1. 이미지 업로드 → 2. 영역 지정(4점 혹은 전체) → 3. 검색 시작")

uploaded = st.file_uploader("자재 이미지 업로드", type=['jpg','png','jpeg'], key=f"up_{st.session_state['uploader_key']}")

if st.sidebar.button("🔄 Reset"):
    for k in ['points','search_done','search_results']: st.session_state[k] = [] if k!='search_done' else False
    st.session_state['uploader_key'] += 1
    st.rerun()

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state['points'] = []; st.session_state['search_done'] = False
        st.session_state['current_img_name'] = uploaded.name
        with st.spinner('📸 모바일 최적화 및 로딩 중...'):
            raw = Image.open(uploaded).convert('RGB')
            # 🚀 [추가] 서버 측 압축: 모바일 업로드 안정성을 위해 즉시 리사이징
            raw.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            st.session_state['proc_img'] = raw
        st.rerun()

    working_img = st.session_state['proc_img']
    
    st.markdown("### 1️⃣ 환경 설정")
    source_type = st.radio("📂 파일 종류", ['📸 현장 사진', '💻 디지털 파일'], horizontal=True)
    is_photo = (source_type == '📸 현장 사진')
    
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1: mat_type = st.selectbox("🧱 자재 종류", ['일반', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)', '석재/콘크리트 (Stone)'], disabled=not is_photo)
    with c_opt2: s_mode = st.radio("🔎 검색 기준", ["🎨 컬러+패턴", "🦓 패턴 중심"], horizontal=True)

    with st.expander("⚙️ 고급 설정", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: lighting = st.selectbox("조명", ['일반/자연광', '백열등', '형광등'], disabled=not is_photo)
        with c2: 
            if st.button("↩️ 90도 회전"): 
                st.session_state['proc_img'] = working_img.rotate(90, expand=True)
                st.session_state['points'] = []; st.rerun()
        with c3:
            bri = st.slider("밝기", 0.5, 2.0, 1.0, 0.1, disabled=not is_photo)
            shp = st.slider("선명도", 0.0, 3.0, 1.5, 0.1, disabled=not is_photo)

    st.markdown("### 2️⃣ 영역 지정")
    col_sel1, col_sel2 = st.columns([3, 2])
    with col_sel1: st.info(f"👇 4곳 클릭 또는 전체 선택 ({len(st.session_state['points'])}/4)")
    with col_sel2:
        if st.button("⏹️ 전체 선택 (Auto)", type="primary"):
            w, h = working_img.size
            st.session_state['points'] = [(0, 0), (w, 0), (w, h), (0, h)]; st.rerun()

    draw_img = working_img.copy(); draw = ImageDraw.Draw(draw_img)
    for i, p in enumerate(st.session_state['points']):
        draw.ellipse((p[0]-10, p[1]-10, p[0]+10, p[1]+10), fill='red', outline='white', width=3)
    if len(st.session_state['points']) == 4:
        draw.polygon([tuple(p) for p in order_points(np.array(st.session_state['points']))], outline='#00FF00', width=5)

    # 🚀 [개선] 좌표 컴포넌트 출력 전 상태 체크로 '지연 현상' 완화
    val = streamlit_image_coordinates(draw_img, key="click_pad")
    if val:
        new_p = (val['x'], val['y'])
        if len(st.session_state['points']) < 4 and (not st.session_state['points'] or st.session_state['points'][-1] != new_p):
            st.session_state['points'].append(new_p); st.rerun()

    if st.session_state['points'] and st.button("❌ 점 지우기"): 
        st.session_state['points'] = []; st.rerun()

    if len(st.session_state['points']) == 4:
        st.markdown("### 3️⃣ 분석")
        warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
        final_img = Image.fromarray(warped)
        if is_photo: final_img = apply_smart_filters(final_img, mat_type, lighting, bri, shp)
        if s_mode == "🦓 패턴 중심": final_img = final_img.convert("L").convert("RGB")
        
        c_p1, c_p2 = st.columns(2)
        with c_p1: st.image(final_img, caption="분석 영역", width=300)
        with c_p2:
            if st.button("🔍 검색 시작", type="primary"):
                with st.spinner('유사 자재 찾는 중...'):
                    x = image.img_to_array(final_img.resize((224, 224)))
                    query_vec = model.predict(preprocess_input(np.expand_dims(x, axis=0)), verbose=0).flatten().reshape(1, -1)
                    db_names = list(feature_db.keys()); db_vecs = np.array(list(feature_db.values()))
                    sims = cosine_similarity(query_vec, db_vecs).flatten()
                    
                    results = []; seen_formal = set()
                    sorted_idx = np.argsort(sims)[::-1]
                    
                    for i in sorted_idx:
                        fname = db_names[i]
                        info = master_map.get(get_digits(fname), {'formal': fname, 'name': '정보 없음', 'lab_no': '-'})
                        f_code = info['formal']
                        
                        # 🚀 [추가] 중복 품번 제거: 이미 상위 결과에 있는 품번은 건너뜀
                        if f_code in seen_formal: continue
                        seen_formal.add(f_code)
                        
                        url_row = df_path[df_path['추출된_품번'].apply(get_digits) == get_digits(fname)]
                        url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                        
                        # 🚀 [개선] 이미지가 있는 경우만 결과에 포함
                        if url:
                            results.append({'formal': f_code, 'name': info['name'], 'lab_no': info['lab_no'], 
                                          'score': sims[i], 'stock': agg_stock.get(get_digits(f_code), 0), 'url': url})
                        if len(results) >= 15: break
                    
                    st.session_state['search_results'] = results; st.session_state['search_done'] = True; st.rerun()

    if st.session_state.get('search_done'):
        st.markdown("---")
        res = st.session_state['search_results']
        def draw_card(item, idx):
            st.markdown(f"**{idx}. {item['formal']}**")
            st.caption(f"{item['name']} (유사도: {item['score']:.1%})")
            st.markdown(f"🔗 [고화질 원본]({item['url']})")
            with st.expander("🖼️ 보기", expanded=False):
                try: st.image(get_direct_url(item['url']), use_container_width=True)
                except: st.write("이미지 로드 실패")
            if item['stock'] >= 100: st.success(f"재고: {item['stock']:,}m")
            else: st.info(f"재고: {item['stock']:,}m")

        t1, t2 = st.tabs(["📊 전체 결과", "✅ 재고 보유 (100m↑)"])
        with t1:
            cols = st.columns(5)
            for i, r in enumerate(res[:10]):
                with cols[i%5]: draw_card(r, i+1)
        with t2:
            hits = [r for r in res if r['stock'] >= 100]
            if hits:
                cols = st.columns(5)
                for i, r in enumerate(hits[:10]):
                    with cols[i%5]: draw_card(r, i+1)
            else: st.warning("100m 이상 재고 없음")
