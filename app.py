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
# --- [1] 유틸리티 및 데이터 매핑 로직 (합리적 로직 적용) ---
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
    return bool(re.match(r'^\d+-\d+-\d+$', str(code).strip()))

def extract_digits(text):
    """4자리 이상 핵심 숫자 추출 (매칭 키)"""
    if pd.isna(text) or str(text).strip() == '-': return ""
    nums = re.findall(r'\d{4,}', str(text))
    return nums[0] if nums else ""

@st.cache_resource
def init_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    if os.path.exists('material_features.pkl'):
        with open('material_features.pkl', 'rb') as f: feature_db = pickle.load(f)
    else:
        st.error("❌ material_features.pkl 파일이 없습니다.")
        st.stop()

    def load_csv(name):
        for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
            try: return pd.read_csv(name, encoding=enc)
            except: continue
        return None

    df_path, df_info, df_stock = load_csv('이미지경로.csv'), load_csv('품목정보.csv'), load_csv('현재고.csv')
    
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
        
        # 매칭 키 생성 (숫자 중심)
        keys = {extract_digits(f), extract_digits(l), f, l}
        for k in keys:
            if not k: continue
            if k not in mapping:
                mapping[k] = info
            else:
                # 🚀 정식 규격(14-...)이 나타나면 기존의 임시 번호 정보를 교체
                if is_formal_code(info['formal']) and not is_formal_code(mapping[k]['formal']):
                    mapping[k] = info
    return mapping

master_map = get_master_map()

# -----------------------------------------------------------
# --- [2] 이미지 분석 및 변환 로직 ---
# -----------------------------------------------------------

def apply_smart_filters(img, category, lighting, brightness, sharpness):
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split()
        b = b.point(lambda i: i * 1.2)
        img = Image.merge('RGB', (r, g, b))
    elif lighting == '형광등 (푸른/녹색 조명)':
        r, g, b = img.split()
        r = r.point(lambda i: i * 1.1)
        img = Image.merge('RGB', (r, g, b))

    enhancer_con = ImageEnhance.Contrast(img)
    enhancer_shp = ImageEnhance.Sharpness(img)
    enhancer_bri = ImageEnhance.Brightness(img)
    
    if category == '마루/우드 (Wood)': img = enhancer_shp.enhance(2.0)
    elif category == '하이그로시/유광 (Glossy)': img = enhancer_con.enhance(1.5)
    
    if brightness != 1.0: img = enhancer_bri.enhance(brightness)
    if sharpness != 1.5: img = enhancer_shp.enhance(sharpness)
    return img

def prepare_image_for_ai(img, mode):
    """이미지 전처리: 대비 정규화로 색상/밝기 편차 극복"""
    img = img.resize((224, 224))
    if mode == "🦓 패턴/질감 중심 (색상 무시)":
        img = img.convert("L").convert("RGB")
    elif mode == "🎨 컬러 중심 (패턴 뭉개기)":
        img = img.filter(ImageFilter.GaussianBlur(radius=15))
    else:
        # 대비 정규화: 밝은 샘플과 어두운 샘플 간의 차이를 줄여줌
        img = ImageOps.autocontrast(img, cutoff=1)
    return img

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w = max(int(np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))), int(np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))))
    h = max(int(np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))), int(np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))

# -----------------------------------------------------------
# --- [3] 메인 UI (기능 완전 복구) ---
# -----------------------------------------------------------

st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색")

st.sidebar.info(f"📅 재고 기준일: {stock_date}")
if st.sidebar.button("🔄 처음부터 다시 하기 (Reset)"):
    for key in ['points', 'search_done', 'search_results', 'raw_img', 'current_img_name']:
        if key in st.session_state: del st.session_state[key]
    st.rerun()

tab1, tab2 = st.tabs(["📂 파일 업로드", "📸 카메라 촬영"])
input_file = None
active_source = None
with tab1:
    uploaded = st.file_uploader("이미지 파일 선택", type=['jpg', 'png', 'jpeg'], key="up")
    if uploaded: input_file, active_source = uploaded, "upload"
with tab2:
    camera_shot = st.camera_input("카메라로 찍기")
    if camera_shot: input_file, active_source = camera_shot, "camera"

if input_file:
    file_id = input_file.name if hasattr(input_file, 'name') else "camera_img"
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != file_id:
        st.session_state['raw_img'] = Image.open(input_file).convert('RGB')
        st.session_state['current_img_name'] = file_id
        st.session_state['points'] = []
        st.session_state['search_done'] = False

    raw = st.session_state['raw_img']
    
    st.markdown("### 1️⃣ 환경 및 검색 설정")
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        source_type = st.radio("📂 원본 종류", ['📸 현장 촬영 사진', '💻 이미지 파일 (스캔/디지털)'], horizontal=True)
        is_photo = (source_type == '📸 현장 촬영 사진')
    with c_set2:
        search_mode = st.radio("🔎 검색 기준", ["🎨 컬러 + 패턴 종합", "🦓 패턴/질감 중심 (색상 무시)", "🎨 컬러 중심 (패턴 뭉개기)"], horizontal=True)

    # 🚀 [복구] 세부 보정 옵션 (Expander)
    with st.expander("⚙️ 세부 보정 및 회전 (조명, 밝기, 선명도)", expanded=is_photo):
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            material_type = st.selectbox("🧱 자재 종류", ['일반 (기본)', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)', '석재/콘크리트 (Stone)'], disabled=not is_photo)
            lighting = st.selectbox("💡 조명 보정", ['일반/자연광', '백열등 (누런 조명)', '형광등 (푸른/녹색 조명)'], disabled=not is_photo)
        with col_ex2:
            st.write("") 
            if st.button("↩️ 사진 90도 회전"):
                st.session_state['raw_img'] = raw.rotate(90, expand=True)
                st.session_state['points'] = []
                st.rerun()
        with col_ex3:
            brightness = st.slider("☀️ 밝기", 0.5, 2.0, 1.0, 0.1, disabled=not is_photo)
            sharpness = st.slider("🔪 선명도", 0.0, 3.0, 1.5, 0.1, disabled=not is_photo)

    st.markdown("### 2️⃣ 영역 지정")
    zoom = st.slider("🔍 이미지 확대/축소", 400, 1500, 700)
    display_img = raw.copy()
    display_img.thumbnail((zoom, zoom))
    
    # 🚀 [복구] 전체 선택 및 Undo 버튼
    col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
    with col_sel1: st.info(f"👇 **모서리 4곳을 클릭**하세요. ({len(st.session_state['points'])}/4)")
    with col_sel2:
        if st.button("⏹️ 이미지 전체 선택", type="primary", use_container_width=True):
            w, h = display_img.size
            st.session_state['points'] = [(0, 0), (w, 0), (w, h), (0, h)]
            st.rerun()
    with col_sel3:
        if st.button("❌ 점 지우기 (Undo)", use_container_width=True):
            st.session_state['points'] = []; st.rerun()
    
    draw = ImageDraw.Draw(display_img)
    for i, p in enumerate(st.session_state['points']):
        draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill='red', outline='white', width=2)
        draw.text((p[0]+12, p[1]-12), str(i+1), fill='red')

    val = streamlit_image_coordinates(display_img, key="roi_click")
    if val:
        new_p = (val['x'], val['y'])
        if len(st.session_state['points']) < 4:
            if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                st.session_state['points'].append(new_p); st.rerun()

    if len(st.session_state['points']) == 4:
        st.markdown("---")
        ratio = raw.width / display_img.width
        pts = np.array(st.session_state['points'], dtype="float32") * ratio
        warped = four_point_transform(np.array(raw), pts)
        final_crop = Image.fromarray(warped)
        
        if is_photo:
            final_crop = apply_smart_filters(final_crop, material_type, lighting, brightness, sharpness)
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1: st.image(final_crop, caption="최종 분석 이미지", width=350)
        with col_res2:
            st.write("✅ 영역 지정 완료. 유사한 자재를 검색하시겠습니까?")
            if st.button("🔍 검색 시작", type="primary", use_container_width=True):
                with st.spinner('AI 분석 중...'):
                    proc_img = prepare_image_for_ai(final_crop, search_mode)
                    x = image.img_to_array(proc_img)
                    x = np.expand_dims(x, axis=0)
                    query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
                    
                    db_names, db_vecs = list(feature_db.keys()), np.array(list(feature_db.values()))
                    sims = cosine_similarity(query_vec, db_vecs).flatten()
                    
                    results = []
                    for i in range(len(db_names)):
                        digits = extract_digits(db_names[i])
                        info = master_map.get(digits, {'formal': db_names[i], 'name': '정보 없음', 'lab_no': '-'})
                        url_match = df_path[df_path['추출된_품번'].apply(extract_digits) == digits]
                        url = url_match.iloc[0]['카카오톡_전송용_URL'] if not url_match.empty else None
                        stock = agg_stock.get(extract_digits(info['formal']), 0)
                        
                        results.append({
                            'formal': info['formal'], 
                            'name': info['name'], 
                            'lab_no': info['lab_no'], 
                            'score': sims[i], 
                            'stock': stock, 
                            'url': url
                        })
                    
                    results.sort(key=lambda x: x['score'], reverse=True)
                    unique_res, seen = [], set()
                    for r in results:
                        if r['formal'] not in seen:
                            unique_res.append(r); seen.add(r['formal'])
                    
                    st.session_state['search_results'] = unique_res[:20]
                    st.session_state['search_done'] = True
                    st.rerun()

# -----------------------------------------------------------
# --- [4] 결과 출력 (KeyError 방지 및 안정화) ---
# -----------------------------------------------------------

if st.session_state.get('search_done'):
    st.markdown("### 🏆 검색 결과")
    results = st.session_state.get('search_results', [])
    
    def display_card(item, idx):
        title = f"{idx}. {item['formal']}"
        if item['lab_no'] != '-' and item['lab_no'] != item['formal']:
            title += f" (Lab: {item['lab_no']})"
        
        st.markdown(f"**{title}**")
        st.write(f"{item['name']}")
        st.caption(f"유사도: {item['score']:.1%}")
        
        if item['url']:
            with st.expander("🖼️ 이미지 확인"):
                try:
                    r = requests.get(get_direct_url(item['url']), timeout=5)
                    st.image(Image.open(BytesIO(r.content)), use_container_width=True)
                except: st.write("로딩 실패")
            st.markdown(f"🔗 [고화질 원본]({item['url']})")
        else: st.write("이미지 없음")
        
        if item['stock'] >= 100: st.success(f"{item['stock']:,}m")
        else: st.info(f"{item['stock']:,}m")

    t1, t2 = st.tabs(["📊 전체 결과", "✅ 재고 보유 (100m↑)"])
    with t1:
        cols = st.columns(5)
        for i, r in enumerate(results[:10]):
            with cols[i % 5]: display_card(r, i+1)
    with t2:
        hits = [r for r in results if r['stock'] >= 100]
        if hits:
            cols = st.columns(5)
            for i, r in enumerate(hits[:10]):
                with cols[i % 5]: display_card(r, i+1)
        else: st.warning("재고 보유 자재 없음")
