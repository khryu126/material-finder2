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
from streamlit_drawable_canvas import st_canvas

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

# --- [2] 투영 변환(Perspective Transform) 로직 ---
def order_points(pts):
    # 좌표 4개를 [좌상, 우상, 우하, 좌하] 순서로 정렬
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 좌상
    rect[2] = pts[np.argmax(s)] # 우하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 우상
    rect[3] = pts[np.argmax(diff)] # 좌하
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 새 이미지의 너비/높이 계산 (최대값 기준)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 투영 변환 행렬 계산 및 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- [3] 이미지 보정 함수 ---
def apply_filters(img, lighting, brightness, sharpness):
    # 조명 보정
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split()
        b = b.point(lambda i: i * 1.2)
        img = Image.merge('RGB', (r, g, b))
    elif lighting == '형광등 (푸른/녹색 조명)':
        r, g, b = img.split()
        r = r.point(lambda i: i * 1.1)
        img = Image.merge('RGB', (r, g, b))
    
    # 밝기/선명도
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
    return img

# --- [4] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색 (투영 보정)")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("자재 이미지를 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'])

if uploaded:
    st.markdown("### 🛠️ 이미지 전처리 및 영역 지정")
    
    with st.expander("📸 촬영 환경 설정", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            lighting = st.selectbox("조명 색상", ['일반/자연광', '백열등 (누런 조명)', '형광등 (푸른/녹색 조명)'])
        with c2:
            brightness = st.slider("💡 밝기", 0.5, 2.0, 1.0, 0.1)
        with c3:
            sharpness = st.slider("🔪 선명도", 0.0, 3.0, 1.5, 0.1)

    # 이미지 로드 및 리사이징 (캔버스용)
    original_image = Image.open(uploaded).convert('RGB')
    
    # 캔버스 크기에 맞게 이미지 리사이징 (너비 600px 고정)
    canvas_width = 600
    w_percent = (canvas_width / float(original_image.size[0]))
    h_size = int((float(original_image.size[1]) * float(w_percent)))
    resized_image = original_image.resize((canvas_width, h_size))
    
    st.info("👇 **이미지 위에서 분석할 영역의 [4개 꼭지점]을 마우스로 클릭하세요.** (순서 상관없음)")
    st.caption("※ 그라데이션이 심한 마루는 **여러 쪽(Plank)을 포함하여 넓게** 찍으세요. 비스듬해도 자동으로 펴줍니다.")

    # 캔버스 생성
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 채우기 색상
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=resized_image,
        update_streamlit=True,
        height=h_size,
        width=canvas_width,
        drawing_mode="polygon", # 다각형 그리기 모드
        key="canvas",
    )

    # 4개 점이 찍혔는지 확인
    pts = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            # 마지막으로 그린 도형의 좌표 가져오기
            path = objects[-1]["path"]
            # path 데이터에서 좌표 추출 (명령어 제외)
            for p in path:
                if p[0] == 'L' or p[0] == 'M': # LineTo or MoveTo
                    pts.append([p[1], p[2]])

    if len(pts) >= 4:
        # 좌표 배열 변환
        pts = np.array(pts[:4], dtype="float32")
        
        # 1. 투영 변환 (Perspective Transform)
        # 리사이즈된 이미지 좌표를 원본 이미지 비율로 복원
        ratio = original_image.size[0] / canvas_width
        original_pts = pts * ratio
        
        # OpenCV 처리를 위해 numpy 변환
        cv_img = np.array(original_image)
        warped = four_point_transform(cv_img, original_pts)
        
        # PIL 이미지로 다시 변환
        final_img = Image.fromarray(warped)
        
        # 2. 조명/선명도 필터 적용
        final_img = apply_filters(final_img, lighting, brightness, sharpness)
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.image(resized_image, caption="원본 (4점 선택)", width=300)
        with c_res2:
            st.image(final_img, caption="보정 결과 (투영 변환 완료)", width=300)

        if st.button("🔍 이 영역으로 검색 시작", type="primary"):
            with st.spinner('분석 중...'):
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

    # 결과 출력
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
            
            if item['stock'] >= 100: st.success(f"재고: {item['stock']:,}m")
            else: st.write(f"재고: {item['stock']:,}m")

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
