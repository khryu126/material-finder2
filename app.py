import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
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

# -----------------------------------------------------------
# --- [1] 유틸리티 및 리소스 ---
# -----------------------------------------------------------

def get_direct_url(url):
    """구글 드라이브 링크를 직접 다운로드 가능한 URL로 변환"""
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: 
        return url
    
    file_id = ""
    if 'file/d/' in url:
        file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    
    if file_id:
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    st.error(f"❌ {target_name} 파일을 찾을 수 없습니다.")
    st.stop()

def extract_digits(text):
    if pd.isna(text) or str(text).strip() == '-': return ""
    text = str(text)
    # 4자리 이상의 숫자 뭉치를 추출 (Lab No 및 품번 핵심 숫자)
    nums = re.findall(r'\d{4,}', text)
    return nums[0] if nums else ""

def is_formal_code(code):
    """정식 품번 형식(예: 14-54130-119)인지 확인하는 로직"""
    if not code or pd.isna(code): return False
    # 하이픈이 두 개 포함된 숫자 위주의 형식을 정식으로 판단
    pattern = r'^\d+-\d+-\d+$'
    return bool(re.match(pattern, str(code).strip()))

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

    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].apply(extract_digits)
    df_stock.loc[df_stock['품번_KEY'] == "", '품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    
    return model, feature_db, df_path, df_info, agg_stock, stock_date

model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f = str(row['상품코드']).strip() if pd.notna(row.get('상품코드')) else ''
        l = str(row.get('Lab No', '')).strip() if pd.notna(row.get('Lab No')) else ''
        n = str(row.get('상품명', '')).strip() if pd.notna(row.get('상품명')) else ''
        
        # 기본 정보 객체
        current_formal = f if f else l
        info = {'formal': current_formal, 'name': n, 'lab_no': l}
        
        # 매핑할 키 후보들 (숫자 및 전체 코드)
        keys = set()
        f_digits = extract_digits(f)
        if f_digits: keys.add(f_digits)
        l_digits = extract_digits(l)
        if l_digits: keys.add(l_digits)
        if f: keys.add(f)
        if l: keys.add(l)
        
        for k in keys:
            if k not in mapping:
                mapping[k] = info
            else:
                # 🚀 [핵심] 기존에 등록된 번호와 비교하여 정식 규격(14-54130-119 등)을 우선순위로 둠
                existing_formal = mapping[k]['formal']
                if is_formal_code(current_formal) and not is_formal_code(existing_formal):
                    mapping[k] = info
    return mapping

master_map = get_master_map()

# -----------------------------------------------------------
# --- [2] 이미지 처리 함수들 ---
# -----------------------------------------------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

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
    enhancer_col = ImageEnhance.Color(img)

    if category == '마루/우드 (Wood)':
        img = enhancer_shp.enhance(2.0)
        img = enhancer_con.enhance(1.1)
    elif category == '하이그로시/유광 (Glossy)':
        img = enhancer_con.enhance(1.5)
        img = enhancer_shp.enhance(1.2)
    elif category == '벽지/패브릭 (Texture)':
        img = enhancer_shp.enhance(1.5)
        img = enhancer_bri.enhance(1.1)
    elif category == '석재/콘크리트 (Stone)':
        img = enhancer_col.enhance(0.8)
        img = enhancer_shp.enhance(1.5)
    
    if brightness != 1.0: img = enhancer_bri.enhance(brightness)
    if sharpness != 1.0: img = enhancer_shp.enhance(sharpness)
        
    return img

def resize_for_display(img, max_width=800):
    if img.width > max_width:
        w_percent = (max_width / float(img.width))
        h_size = int((float(img.height) * float(w_percent)))
        return img.resize((max_width, h_size), Image.Resampling.LANCZOS)
    return img

# -----------------------------------------------------------
# --- [3] 메인 UI 레이아웃 ---
# -----------------------------------------------------------

st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

if 'points' not in st.session_state: st.session_state['points'] = []
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0
if 'search_done' not in st.session_state: st.session_state['search_done'] = False

tab1, tab2 = st.tabs(["📂 파일 업로드", "📸 카메라 촬영"])

input_file = None
active_source = None

with tab1:
    uploaded = st.file_uploader("이미지 파일 선택", type=['jpg', 'png', 'tif', 'jpeg'], key=f"up_{st.session_state['uploader_key']}")
    if uploaded:
        input_file = uploaded
        active_source = "upload"

with tab2:
    camera_shot = st.camera_input("카메라로 찍기")
    if camera_shot:
        input_file = camera_shot
        active_source = "camera"

if st.sidebar.button("🔄 처음부터 다시 하기 (Reset)"):
    st.session_state['points'] = []
    st.session_state['search_done'] = False
    st.session_state['search_results'] = None
    st.session_state['uploader_key'] += 1
    st.session_state['proc_img'] = None
    st.session_state['current_img_name'] = None
    st.rerun()

if input_file:
    is_new = False
    file_id = input_file.name if hasattr(input_file, 'name') else "camera_img"
    
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != file_id:
        is_new = True

    if is_new:
        st.session_state['points'] = []
        st.session_state['search_done'] = False
        st.session_state['search_results'] = None
        st.session_state['current_img_name'] = file_id
        
        with st.spinner('📸 이미지 최적화 중...'):
            raw = Image.open(input_file).convert('RGB')
            st.session_state['raw_img'] = raw
            st.session_state['proc_img'] = resize_for_display(raw, max_width=800)
        st.rerun()

    if 'raw_img' in st.session_state:
        working_raw = st.session_state['raw_img']

        st.markdown("### 1️⃣ 환경 설정")
        source_type = st.radio("📂 원본 종류", ['📸 현장 촬영 사진', '💻 이미지 파일 (스캔/디지털)'], index=0, horizontal=True)
        is_photo = (source_type == '📸 현장 촬영 사진')

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            material_type = st.selectbox("🧱 자재 종류", ['일반 (기본)', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)', '석재/콘크리트 (Stone)'], disabled=not is_photo)
        with col_opt2:
            search_mode = st.radio("🔎 검색 기준", ["🎨 컬러 + 패턴 (기본)", "🦓 패턴/질감 중심 (흑백)", "🎨 컬러/톤 중심 (패턴 뭉개기)"], horizontal=True)

        st.markdown("### 2️⃣ 영역 지정")
        zoom_level = st.slider("🔍 이미지 확대/축소", 300, 1500, 600, 50)
        display_img = resize_for_display(working_raw, max_width=zoom_level)

        col_sel1, col_sel2 = st.columns([3, 2])
        with col_sel1: st.info(f"👇 **모서리 4곳을 클릭**하세요. ({len(st.session_state['points'])}/4)")
        with col_sel2:
            if st.button("⏹️ 전체 선택 (스캔파일용)", type="primary"):
                w, h = display_img.size
                st.session_state['points'] = [(0, 0), (w, 0), (w, h), (0, h)]
                st.rerun()

        draw_img = display_img.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, p in enumerate(st.session_state['points']):
            draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill='red', outline='white', width=2)
            draw.text((p[0]+10, p[1]-10), str(i+1), fill='red')

        if len(st.session_state['points']) == 4:
            pts = np.array(st.session_state['points'])
            rect = order_points(pts)
            draw.polygon([tuple(p) for p in rect], outline='#00FF00', width=4)

        value = streamlit_image_coordinates(draw_img, key=f"click_pad_{zoom_level}")

        if value is not None:
            new_point = (value['x'], value['y'])
            if len(st.session_state['points']) < 4:
                if not st.session_state['points'] or st.session_state['points'][-1] != new_point:
                    st.session_state['points'].append(new_point)
                    st.rerun()

        if len(st.session_state['points']) == 4:
            st.markdown("### 3️⃣ 분석 결과")
            ratio = working_raw.width / display_img.width
            original_pts = np.array(st.session_state['points'], dtype="float32") * ratio
            
            cv_img = np.array(working_raw)
            warped = four_point_transform(cv_img, original_pts)
            final_img = Image.fromarray(warped)
            
            if is_photo:
                final_img = apply_smart_filters(final_img, material_type, '일반/자연광', 1.0, 1.5)
            
            proc_img_for_ai = final_img.copy()
            if search_mode == "🦓 패턴/질감 중심 (흑백)": proc_img_for_ai = final_img.convert("L").convert("RGB")
            elif search_mode == "🎨 컬러/톤 중심 (패턴 뭉개기)": proc_img_for_ai = final_img.filter(ImageFilter.GaussianBlur(radius=10))

            st.image(final_img, caption="분석 대상 이미지", width=300)
            
            if st.button("🔍 유사 자재 검색 시작", type="primary"):
                with st.spinner('유사한 자재 찾는 중...'):
                    x = image.img_to_array(proc_img_for_ai.resize((224, 224)))
                    x = np.expand_dims(x, axis=0)
                    query_vec = model.predict(preprocess_input(x), verbose=0).flatten().reshape(1, -1)
                    
                    db_names, db_vecs = list(feature_db.keys()), np.array(list(feature_db.values()))
                    sims = cosine_similarity(query_vec, db_vecs).flatten()
                    
                    raw_results = []
                    for i in range(len(db_names)):
                        fname = db_names[i]
                        target_digits = extract_digits(fname)
                        info = master_map.get(target_digits)
                        
                        if not info:
                            info = {'formal': fname, 'name': '정보 없음', 'lab_no': '-'}

                        qty = agg_stock.get(extract_digits(info['formal']), 0)
                        
                        url_match = df_path[df_path['추출된_품번'].apply(extract_digits) == target_digits]
                        url = url_match.iloc[0]['카카오톡_전송용_URL'] if not url_match.empty else None
                        
                        raw_results.append({'formal': info['formal'], 'name': info['name'], 'lab_no': info['lab_no'], 'score': sims[i], 'stock': qty, 'url': url})
                    
                    raw_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    seen_codes, unique_results = set(), []
                    for res in raw_results:
                        if res['formal'] not in seen_codes:
                            unique_results.append(res)
                            seen_codes.add(res['formal'])
                    
                    st.session_state['search_results'] = unique_results
                    st.session_state['search_done'] = True
                    st.rerun()

# -----------------------------------------------------------
# --- [4] 결과 표시 (이미지 링크 수정 반영) ---
# -----------------------------------------------------------

if st.session_state.get('search_done'):
    st.markdown("---")
    results = st.session_state['search_results']

    def display_card(item, idx):
        title_text = f"{idx}. {item['formal']}"
        if item['lab_no'] != '-' and item['lab_no'] != item['formal']:
            title_text += f" (Lab: {item['lab_no']})"
        st.markdown(f"**{title_text}**")
        st.write(f"{item['name']}")
        st.caption(f"유사도: {item['score']:.1%}")
        
        if item['url']:
            direct_url = get_direct_url(item['url'])
            st.markdown(f"🔗 [**고화질 원본**]({item['url']})")
            with st.expander("🖼️ 이미지 보기", expanded=False):
                try:
                    # 구글 드라이브 보안 이슈로 인해 requests로 데이터를 받아와서 표시
                    resp = requests.get(direct_url, timeout=5)
                    st.image(Image.open(BytesIO(resp.content)), use_container_width=True)
                except:
                    st.warning("이미지를 불러올 수 없습니다. 위 링크를 클릭하세요.")
        else:
            st.write("이미지 없음")
        
        stock_text = f"{item['stock']:,}m"
        if item['stock'] >= 100: st.success(stock_text)
        else: st.write(stock_text)

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
