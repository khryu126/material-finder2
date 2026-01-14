import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------
# 🚑 [시스템 패치] Streamlit 이미지 렌더링 호환성 해결
# -----------------------------------------------------------
import streamlit.elements.image as st_image

def local_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="auto", image_id=None):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

if not hasattr(st_image, 'image_to_url'):
    st_image.image_to_url = local_image_to_url

# -----------------------------------------------------------
# --- [1] 유틸리티 및 데이터 매핑 로직 ---
# -----------------------------------------------------------

def get_direct_url(url):
    """구글 드라이브 URL 변환 (안정적 다운로드 링크)"""
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: 
        return url
    file_id = ""
    if 'file/d/' in url: file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url: file_id = url.split('id=')[1].split('&')[0]
    return f'https://drive.google.com/uc?export=download&id={file_id}' if file_id else url

def is_formal_code(code):
    """정식 품번(14-54130-119 등) 형식 검사"""
    if not code or pd.isna(code): return False
    pattern = r'^\d+-\d+-\d+$' # 숫자-숫자-숫자 형태
    return bool(re.match(pattern, str(code).strip()))

def extract_digits(text):
    """4자리 이상 핵심 숫자 추출 (매칭 키)"""
    if pd.isna(text) or str(text).strip() == '-': return ""
    nums = re.findall(r'\d{4,}', str(text))
    return nums[0] if nums else ""

@st.cache_resource
def init_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # 특징값 DB 로드
    if os.path.exists('material_features.pkl'):
        with open('material_features.pkl', 'rb') as f:
            feature_db = pickle.load(f)
    else:
        st.error("❌ material_features.pkl 파일이 없습니다.")
        st.stop()

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
    df_stock['품번_KEY'] = df_stock['품번'].apply(extract_digits)
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    
    return model, feature_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

@st.cache_data
def get_master_map():
    """품번 우선순위 적용 매핑 (정식 규격 우선)"""
    mapping = {}
    for _, row in df_info.iterrows():
        f = str(row.get('상품코드', '')).strip()
        l = str(row.get('Lab No', '')).strip()
        n = str(row.get('상품명', '')).strip()
        
        info = {'formal': f if f else l, 'name': n, 'lab_no': l}
        
        keys = {extract_digits(f), extract_digits(l), f, l}
        for k in keys:
            if not k: continue
            if k not in mapping:
                mapping[k] = info
            else:
                # 🚀 정식 규격이 나타나면 임시 번호 정보를 덮어씌움
                if is_formal_code(info['formal']) and not is_formal_code(mapping[k]['formal']):
                    mapping[k] = info
    return mapping

master_map = get_master_map()

# -----------------------------------------------------------
# --- [2] 이미지 분석 및 변환 로직 ---
# -----------------------------------------------------------

def prepare_image_for_ai(img, mode):
    """이미지 전처리: 대비 정규화로 밝기 차이 극복"""
    img = img.resize((224, 224))
    if mode == "🦓 패턴 중심(흑백)":
        img = img.convert("L").convert("RGB")
    elif mode == "🎨 컬러 중심(블러)":
        img = img.filter(ImageFilter.GaussianBlur(radius=15))
    else:
        # 밝기/대비 정규화를 통해 색상 톤 오차 감소
        img = ImageOps.autocontrast(img, cutoff=1)
    return img

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(int(np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))), int(np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))))
    height = max(int(np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))), int(np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

# -----------------------------------------------------------
# --- [3] 메인 UI (영역 지정 복구) ---
# -----------------------------------------------------------

st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색")

# 사이드바 정보
st.sidebar.info(f"📅 재고 기준일: {stock_date}")
if st.sidebar.button("🔄 처음부터 다시 하기 (Reset)"):
    for key in ['points', 'search_done', 'search_results', 'raw_img']:
        if key in st.session_state: del st.session_state[key]
    st.rerun()

tab1, tab2 = st.tabs(["📂 파일 업로드", "📸 카메라 촬영"])
input_file = None
with tab1:
    uploaded = st.file_uploader("이미지 파일", type=['jpg', 'png', 'jpeg'], key="up")
    if uploaded: input_file = uploaded
with tab2:
    camera_shot = st.camera_input("카메라 촬영")
    if camera_shot: input_file = camera_shot

if input_file:
    if 'raw_img' not in st.session_state:
        st.session_state['raw_img'] = Image.open(input_file).convert('RGB')
        st.session_state['points'] = []

    raw = st.session_state['raw_img']
    
    st.markdown("### 1️⃣ 환경 및 검색 설정")
    c1, c2 = st.columns(2)
    with c1: source_type = st.radio("📂 원본 종류", ['📸 현장 촬영', '💻 디지털 스캔'], horizontal=True)
    with c2: search_mode = st.radio("🔎 검색 기준", ["🎨 컬러+패턴", "🦓 패턴 중심(흑백)", "🎨 컬러 중심(블러)"], horizontal=True)

    st.markdown("### 2️⃣ 영역 지정")
    zoom = st.slider("🔍 이미지 확대/축소", 400, 1200, 700)
    display_img = raw.copy()
    display_img.thumbnail((zoom, zoom))
    
    # 🚀 [복구] 점 지우기 및 안내 섹션
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        st.info(f"👇 **모서리 4곳을 클릭**하세요. ({len(st.session_state['points'])}/4)")
    with col_sel2:
        if st.button("❌ 점 지우기 (Undo)", use_container_width=True):
            st.session_state['points'] = []
            st.rerun()
    
    # 포인트 캔버스
    draw = ImageDraw.Draw(display_img)
    for i, p in enumerate(st.session_state['points']):
        draw.ellipse((p[0]-6, p[1]-6, p[0]+6, p[1]+6), fill='red', outline='white', width=2)
        draw.text((p[0]+10, p[1]-10), str(i+1), fill='red')
    
    val = streamlit_image_coordinates(display_img, key="roi_click")
    if val:
        new_p = (val['x'], val['y'])
        if len(st.session_state['points']) < 4:
            if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                st.session_state['points'].append(new_p)
                st.rerun()

    # 4개 점이 모두 찍히면 분석 준비
    if len(st.session_state['points']) == 4:
        st.markdown("---")
        ratio = raw.width / display_img.width
        pts = np.array(st.session_state['points'], dtype="float32") * ratio
        warped = four_point_transform(np.array(raw), pts)
        final_crop = Image.fromarray(warped)
        
        col_crop1, col_crop2 = st.columns([1, 2])
        with col_crop1:
            st.image(final_crop, caption="잘라낸 자재 이미지", width=300)
        with col_crop2:
            st.write("✅ 영역 지정 완료. 검색을 시작하세요.")
            if st.button("🔍 유사 자재 검색 시작", type="primary", use_container_width=True):
                with st.spinner('유사한 자재를 찾는 중...'):
                    proc_img = prepare_image_for_ai(final_crop, search_mode)
                    x = image.img_to_array(proc_img)
                    x = np.expand_dims(x, axis=0)
                    query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
                    
                    db_names, db_vecs = list(feature_db.keys()), np.array(list(feature_db.values()))
                    sims = cosine_similarity(query_vec, db_vecs).flatten()
                    
                    raw_res = []
                    for i in range(len(db_names)):
                        digits = extract_digits(db_names[i])
                        info = master_map.get(digits, {'formal': db_names[i], 'name': '정보없음', 'lab_no': '-'})
                        
                        url_match = df_path[df_path['추출된_품번'].apply(extract_digits) == digits]
                        url = url_match.iloc[0]['카카오톡_전송용_URL'] if not url_match.empty else None
                        stock = agg_stock.get(extract_digits(info['formal']), 0)
                        
                        raw_res.append({'info': info, 'score': sims[i], 'stock': stock, 'url': url})
                    
                    raw_res.sort(key=lambda x: x['score'], reverse=True)
                    unique_res, seen = [], set()
                    for r in raw_res:
                        if r['info']['formal'] not in seen:
                            unique_res.append(r); seen.add(r['info']['formal'])
                    
                    st.session_state['search_results'] = unique_res[:20]
                    st.session_state['search_done'] = True
                    st.rerun()

# -----------------------------------------------------------
# --- [4] 결과 출력 ---
# -----------------------------------------------------------

if st.session_state.get('search_done'):
    st.markdown("### 🏆 검색 결과 (상위 20개)")
    results = st.session_state['search_results']
    
    cols = st.columns(5)
    for i, item in enumerate(results):
        with cols[i % 5]:
            info = item['info']
            st.markdown(f"**{i+1}. {info['formal']}**")
            if info['lab_no'] != '-' and info['lab_no'] != info['formal']:
                st.caption(f"(Lab: {info['lab_no']})")
            st.write(f"{info['name']}")
            st.caption(f"유사도: {item['score']:.1%}")
            
            if item['url']:
                with st.expander("🖼️ 이미지 확인"):
                    try:
                        r = requests.get(get_direct_url(item['url']), timeout=5)
                        st.image(Image.open(BytesIO(r.content)), use_container_width=True)
                    except: st.write("로딩 실패")
                st.markdown(f"🔗 [원본 링크]({item['url']})")
            
            if item['stock'] >= 100: st.success(f"{item['stock']:,}m")
            else: st.info(f"{item['stock']:,}m")
