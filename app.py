import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
from PIL import Image, ImageEnhance, ImageDraw, ImageOps
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# -----------------------------------------------------------
# 🚑 [필수 패치] Streamlit 호환성 해결
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

# --- [1] 유틸리티 ---
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

# --- [2] 이미지 처리 (투영/보정) ---
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

def apply_filters(img, lighting, surface, flooring_mode, brightness, sharpness):
    # 조명 보정
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

    if flooring_mode != '해당 없음':
        img = enhancer_shp.enhance(2.0)
        img = enhancer_con.enhance(1.1)
    else:
        if surface == '하이그로시 (반사 심함)':
            img = enhancer_con.enhance(1.5)
        elif surface == '매트/엠보 (무광)':
            img = enhancer_con.enhance(1.2)
        if sharpness != 1.0:
            img = enhancer_shp.enhance(sharpness)
    
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    return img

def resize_image_for_speed(img, max_width=800):
    if img.width > max_width:
        w_percent = (max_width / float(img.width))
        h_size = int((float(img.height) * float(w_percent)))
        return img.resize((max_width, h_size), Image.Resampling.LANCZOS)
    return img

# --- [3] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

# 세션 상태
if 'points' not in st.session_state: st.session_state['points'] = []
if 'current_img' not in st.session_state: st.session_state['current_img'] = None
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0

uploaded = st.file_uploader("자재 이미지를 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'], key=f"up_{st.session_state['uploader_key']}")

if st.sidebar.button("🔄 처음부터 다시 하기"):
    st.session_state['points'] = []
    st.session_state['current_img'] = None
    st.session_state['uploader_key'] += 1
    st.rerun()

if uploaded:
    if st.session_state['current_img'] is None or uploaded.name != st.session_state.get('last_filename'):
        try:
            raw = Image.open(uploaded).convert('RGB')
            st.session_state['current_img'] = resize_image_for_speed(raw, max_width=800)
            st.session_state['last_filename'] = uploaded.name
            st.session_state['points'] = []
        except:
            st.error("이미지 로딩 실패")
            st.stop()

    working_img = st.session_state['current_img']

    st.markdown("### 🛠️ 검색 설정 및 영역 지정")
    
    # [NEW] 검색 모드 추가
    search_mode = st.radio(
        "🔎 검색 기준 선택", 
        ["🎨 컬러 + 패턴 종합 (기본)", "🦓 패턴/질감 중심 (색상 무시)"], 
        horizontal=True,
        help="조명 색이 너무 강하거나, 색상은 다르지만 무늬가 같은 자재를 찾을 때 '패턴 중심'을 선택하세요."
    )

    with st.expander("📸 상세 환경 설정 (조명/재질)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            source_type = st.radio("원본 종류", ['사진 촬영본', '이미지 파일 (스캔/디지털)'])
        with c2:
            lighting = st.selectbox("조명 색상", ['일반/자연광', '백열등 (누런 조명)', '형광등 (푸른/녹색 조명)'], disabled=(source_type!='사진 촬영본'))
        with c3:
            surface = st.selectbox("표면 재질", ['일반', '하이그로시 (반사 심함)', '매트/엠보 (무광)'], disabled=(source_type!='사진 촬영본'))
        with c4:
            flooring_mode = st.selectbox("마루 모드", ['해당 없음', '일반 마루', '헤링본/쉐브론'], disabled=(source_type!='사진 촬영본'))
        
        c5, c6, c7 = st.columns(3)
        with c5:
            if st.button("↩️ 90도 회전"):
                st.session_state['current_img'] = working_img.rotate(90, expand=True)
                st.session_state['points'] = []
                st.rerun()
        with c6:
            brightness = st.slider("💡 밝기", 0.5, 2.0, 1.0, 0.1) if source_type == '사진 촬영본' else 1.0
        with c7:
            sharpness = st.slider("🔪 선명도", 0.0, 3.0, 1.5, 0.1) if source_type == '사진 촬영본' else 1.0

    # 좌표 그리기
    draw_img = working_img.copy()
    draw = ImageDraw.Draw(draw_img)
    for p in st.session_state['points']:
        draw.ellipse((p[0]-10, p[1]-10, p[0]+10, p[1]+10), fill='red', outline='white')
    
    if len(st.session_state['points']) == 4:
        pts = np.array(st.session_state['points'])
        rect = order_points(pts)
        draw.polygon([tuple(p) for p in rect], outline='red', width=5)

    st.info(f"👇 **자재의 모서리 4곳을 클릭하세요.** ({len(st.session_state['points'])}/4 완료)")
    
    value = streamlit_image_coordinates(draw_img, key="pilot")

    if value is not None:
        point = (value['x'], value['y'])
        if not st.session_state['points'] or st.session_state['points'][-1] != point:
            if len(st.session_state['points']) < 4:
                st.session_state['points'].append(point)
                st.rerun()

    if len(st.session_state['points']) > 0:
        if st.button("❌ 점 다시 찍기"):
            st.session_state['points'] = []
            st.rerun()

    # 분석 시작
    if len(st.session_state['points']) == 4:
        pts = np.array(st.session_state['points'], dtype="float32")
        cv_img = np.array(working_img)
        warped = four_point_transform(cv_img, pts)
        
        final_img = Image.fromarray(warped)
        if source_type == '사진 촬영본':
            final_img = apply_filters(final_img, lighting, surface, flooring_mode, brightness, sharpness)
        
        # [NEW] 패턴 중심 모드일 경우 흑백 변환 (색상 정보 제거)
        if search_mode == "🦓 패턴/질감 중심 (색상 무시)":
            final_img = final_img.convert("L").convert("RGB")
            st.caption("ℹ️ 색상을 제거하고 텍스처 위주로 분석합니다.")

        st.success("✅ 준비 완료!")
        st.image(final_img, caption="AI가 분석할 이미지", width=300)

        if st.button("🔍 검색 시작", type="primary"):
            with st.spinner('AI 분석 중...'):
                x = image.img_to_array(final_img.resize((224, 224)))
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
