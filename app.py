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
# 🚑 [필수 패치] Streamlit 호환성 해결 (흰 화면 방지)
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

# --- [1] 유틸리티 및 리소스 로드 ---
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

# --- [2] 이미지 처리 (투영 & 스마트 필터) ---
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
    # 1. 조명 보정
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

    # 2. 자재별 자동 보정 (사용자가 고른 카테고리 기반)
    if category == '마루/우드 (Wood)':
        # 나뭇결 강조: 선명도 대폭 증가, 대비 약간 증가
        img = enhancer_shp.enhance(2.0)
        img = enhancer_con.enhance(1.1)
    elif category == '하이그로시/유광 (Glossy)':
        # 빛반사 제거: 대비 대폭 증가 (반사광 날림)
        img = enhancer_con.enhance(1.5)
        img = enhancer_shp.enhance(1.2)
    elif category == '벽지/패브릭 (Texture)':
        # 질감 강조: 적당한 선명도
        img = enhancer_shp.enhance(1.5)
        img = enhancer_bri.enhance(1.1)
    elif category == '석재/콘크리트 (Stone)':
        # 색상 왜곡 방지: 채도 감소, 선명도 증가
        img = enhancer_col.enhance(0.8)
        img = enhancer_shp.enhance(1.5)
    
    # 3. 사용자 수동 미세 조정
    if brightness != 1.0: img = enhancer_bri.enhance(brightness)
    if sharpness != 1.0: img = enhancer_shp.enhance(sharpness)
        
    return img

def resize_for_display(img, max_width=800):
    if img.width > max_width:
        w_percent = (max_width / float(img.width))
        h_size = int((float(img.height) * float(w_percent)))
        return img.resize((max_width, h_size), Image.Resampling.LANCZOS)
    return img

# --- [3] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

# --- 세션 상태 관리 ---
if 'points' not in st.session_state: st.session_state['points'] = []
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0
if 'search_done' not in st.session_state: st.session_state['search_done'] = False

# --- 가이드라인 (Expander) ---
with st.expander("📘 [필독] 사용 방법 및 옵션 가이드 (클릭)", expanded=False):
    st.markdown("""
    **1. 사진 촬영 및 업로드**
    * 최대한 정면에서 찍으면 좋지만, **비스듬하게 서서 찍어도 괜찮습니다.** (4점 보정 기능이 펴줍니다!)
    * **고화질 사진**도 자동으로 최적화되므로 그냥 업로드하세요.

    **2. 자재 종류 선택 (중요 ⭐)**
    * **마루/우드:** 나뭇결이 흐릿할 때 선택하세요. 선명도를 확 올려서 무늬를 잡아냅니다.
    * **하이그로시:** 빛 반사가 심해서 하얗게 뜬 부분이 많을 때 선택하세요.
    * **석재/콘크리트:** 돌 표면의 거친 질감을 찾을 때 유리합니다.

    **3. 영역 지정 (4점 콕콕)**
    * 마우스로 자재의 **모서리 4군데를 클릭**하세요.
    * 4번째 점을 찍는 순간, 찌그러진 사진이 **네모 반듯하게** 펴집니다.
    
    **4. 검색 모드**
    * **패턴 중심(흑백):** "색깔은 달라도 되니 무늬가 똑같은 걸 찾아줘!" (추천 👍)
    * **컬러+패턴:** "색깔도 비슷해야 해!" (우드 톤 구분할 때)
    """)

# --- 업로더 및 초기화 ---
uploaded = st.file_uploader("자재 이미지를 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'], key=f"up_{st.session_state['uploader_key']}")

# 이미지 리셋 버튼
if st.sidebar.button("🔄 처음부터 다시 하기 (Reset)"):
    st.session_state['points'] = []
    st.session_state['search_done'] = False
    st.session_state['search_results'] = None
    st.session_state['uploader_key'] += 1
    st.rerun()

if uploaded:
    # 🧹 [자동 리셋] 새로운 파일이 들어오면 기존 결과/좌표 싹 지우기
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state['points'] = []
        st.session_state['search_done'] = False
        st.session_state['search_results'] = None
        st.session_state['current_img_name'] = uploaded.name
        
        # ⏳ [로딩 표시] 대용량 이미지 처리 중 사용자 안심시키기
        with st.spinner('📸 고화질 이미지를 분석용으로 최적화하는 중입니다... 잠시만요!'):
            try:
                raw = Image.open(uploaded).convert('RGB')
                st.session_state['proc_img'] = resize_for_display(raw, max_width=800)
            except:
                st.error("이미지를 처리할 수 없습니다.")
                st.stop()

    working_img = st.session_state['proc_img']

    # --- 옵션 설정 UI ---
    st.markdown("### 1️⃣ 환경 설정")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        material_type = st.selectbox(
            "🧱 자재 종류 (자동 필터)", 
            ['일반 (기본)', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)', '석재/콘크리트 (Stone)'],
            help="자재 특성에 맞춰 AI가 더 잘 볼 수 있도록 이미지를 자동 보정합니다."
        )
    with col_opt2:
        search_mode = st.radio(
            "🔎 검색 기준", 
            ["🎨 컬러 + 패턴 종합", "🦓 패턴/질감 중심 (색상 무시)"], 
            horizontal=True,
            help="조명 때문에 색이 이상하게 찍혔다면 '패턴 중심'을 선택하세요. 흑백으로 변환하여 무늬만 비교합니다."
        )

    with st.expander("⚙️ 고급 설정 (조명, 회전, 밝기)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            lighting = st.selectbox("조명 색상", ['일반/자연광', '백열등 (누런 조명)', '형광등 (푸른/녹색 조명)'], help="현장 조명이 너무 노랗거나 푸르다면 선택하세요.")
        with c2:
            if st.button("↩️ 사진 90도 회전"):
                st.session_state['proc_img'] = working_img.rotate(90, expand=True)
                st.session_state['points'] = [] # 회전하면 좌표 초기화
                st.rerun()
        with c3:
            brightness = st.slider("밝기 조절", 0.5, 2.0, 1.0, 0.1, help="사진이 너무 어두우면 밝게, 너무 밝으면 어둡게 조절하세요.")
            sharpness = st.slider("선명도 조절", 0.0, 3.0, 1.5, 0.1, help="무늬가 흐릿하면 선명도를 높이세요.")

    # --- 좌표 찍기 ---
    st.markdown("### 2️⃣ 영역 지정 (4점 클릭)")
    st.info(f"👇 이미지에서 분석할 자재의 **모서리 4곳을 클릭**해주세요. ({len(st.session_state['points'])}/4 완료)")
    
    draw_img = working_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # 점 그리기
    for i, p in enumerate(st.session_state['points']):
        draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill='red', outline='white', width=2)
        draw.text((p[0]+10, p[1]-10), str(i+1), fill='red')

    # 4점 완성 시 선 그리기
    if len(st.session_state['points']) == 4:
        pts = np.array(st.session_state['points'])
        rect = order_points(pts)
        draw.polygon([tuple(p) for p in rect], outline='#00FF00', width=4)

    # 인터랙티브 이미지
    value = streamlit_image_coordinates(draw_img, key="click_pad")

    if value is not None:
        new_point = (value['x'], value['y'])
        if len(st.session_state['points']) < 4:
            if not st.session_state['points'] or st.session_state['points'][-1] != new_point:
                st.session_state['points'].append(new_point)
                st.rerun()

    if len(st.session_state['points']) > 0 and len(st.session_state['points']) < 4:
        if st.button("❌ 점 취소하고 다시 찍기"):
            st.session_state['points'] = []
            st.rerun()

    # --- 분석 및 결과 ---
    if len(st.session_state['points']) == 4:
        st.markdown("### 3️⃣ 분석 결과")
        
        pts = np.array(st.session_state['points'], dtype="float32")
        cv_img = np.array(working_img)
        warped = four_point_transform(cv_img, pts)
        
        final_img = Image.fromarray(warped)
        
        # [스마트 필터 적용]
        final_img = apply_smart_filters(final_img, material_type, lighting, brightness, sharpness)
        
        # [패턴 모드일 경우 흑백 변환]
        if search_mode == "🦓 패턴/질감 중심 (색상 무시)":
            final_img = final_img.convert("L").convert("RGB")

        col_prev1, col_prev2 = st.columns(2)
        with col_prev1: 
            st.image(final_img, caption="AI가 분석할 최종 이미지", width=300)
        with col_prev2:
            st.write("👉 이미지가 잘 펴졌나요?")
            if st.button("🔍 이대로 검색 시작", type="primary"):
                with st.spinner('AI가 데이터베이스를 뒤지는 중입니다...'):
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
                    st.rerun() # 결과창 바로 띄우기 위해 리런

    # 결과 표시
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
