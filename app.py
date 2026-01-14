import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates # 가볍고 확실한 좌표 라이브러리

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

# --- [2] 투영 변환 로직 (좌표 4개 받아서 펴기) ---
def order_points(pts):
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
    # 조명
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

    # 재질/마루
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
    
    # 밝기
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
        
    return img

# --- [3] UI 구성 ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색 (4점 클릭)")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

# 세션 상태 초기화 (클릭 좌표 저장용)
if 'points' not in st.session_state:
    st.session_state['points'] = []
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

# 이미지 업로더 (키를 바꿔서 강제 리셋 가능하게 함)
uploaded = st.file_uploader("자재 이미지를 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'], key=f"uploader_{st.session_state['uploader_key']}")

# 이미지 리셋 버튼
if st.sidebar.button("🔄 이미지/좌표 초기화"):
    st.session_state['points'] = []
    st.session_state['uploader_key'] += 1 # 업로더 초기화
    st.rerun()

if uploaded:
    # 이미지가 바뀌면 좌표 초기화
    if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded.name:
        st.session_state['points'] = []
        st.session_state['last_uploaded'] = uploaded.name

    st.markdown("### 🛠️ 촬영 환경 및 영역 지정")
    
    with st.expander("📸 환경 설정 (조명/재질 등)", expanded=True):
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
            # 회전: 캔버스가 아니므로 즉시 적용해서 보여줌
            rotation = st.radio("사진 회전", [0, 90, 180, 270], horizontal=True, format_func=lambda x: f"↩️ {x}도" if x else "원본")
        with c6:
            brightness = st.slider("💡 밝기", 0.5, 2.0, 1.0, 0.1) if source_type == '사진 촬영본' else 1.0
        with c7:
            sharpness = st.slider("🔪 선명도", 0.0, 3.0, 1.5, 0.1) if source_type == '사진 촬영본' else 1.0

    # 1. 원본 이미지 로드 및 전처리 (회전만 적용)
    try:
        raw_img = Image.open(uploaded).convert('RGB')
        if rotation != 0:
            raw_img = raw_img.rotate(-rotation, expand=True)
    except:
        st.error("이미지 로딩 실패")
        st.stop()

    # 2. 화면 표시용 리사이징
    # (너무 크면 좌표 클릭이 불편하므로 너비 600px로 고정)
    disp_width = 600
    w_percent = (disp_width / float(raw_img.size[0]))
    disp_height = int((float(raw_img.size[1]) * float(w_percent)))
    disp_img = raw_img.resize((disp_width, disp_height))

    # 3. 클릭된 점 그리기 (시각적 피드백)
    # disp_img 위에 빨간 점을 그려서 보여줍니다.
    draw_img = disp_img.copy()
    draw = ImageDraw.Draw(draw_img)
    for p in st.session_state['points']:
        # 반지름 5px 빨간 원
        draw.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill='red', outline='white')
        
    # 점 4개가 되면 선으로 이어줌 (사각형 미리보기)
    if len(st.session_state['points']) == 4:
        pts = np.array(st.session_state['points'])
        # 순서 정렬 (좌상, 우상, 우하, 좌하)
        rect = order_points(pts)
        draw.polygon([tuple(p) for p in rect], outline='red', width=3)

    # 4. 좌표 입력 컴포넌트 (이미지 클릭 감지)
    st.info(f"👇 **자재의 모서리 4곳을 클릭하세요.** ({len(st.session_state['points'])}/4 완료)")
    
    # 여기서 클릭하면 좌표가 반환됩니다.
    value = streamlit_image_coordinates(draw_img, key="pilot")

    # 클릭 이벤트 처리
    if value is not None:
        point = (value['x'], value['y'])
        # 중복 클릭 방지 (같은 위치 연속 클릭 무시)
        if not st.session_state['points'] or st.session_state['points'][-1] != point:
            if len(st.session_state['points']) < 4:
                st.session_state['points'].append(point)
                st.rerun() # 점 찍었으니 화면 갱신해서 빨간 점 보여주기

    # 좌표 초기화 버튼 (잘못 찍었을 때)
    if len(st.session_state['points']) > 0:
        if st.button("❌ 점 다시 찍기"):
            st.session_state['points'] = []
            st.rerun()

    # 5. 분석 시작 (4점 완료 시)
    if len(st.session_state['points']) == 4:
        # 화면 좌표(600px 기준)를 원본 이미지 비율로 변환
        ratio = raw_img.size[0] / disp_width
        original_pts = np.array(st.session_state['points'], dtype="float32") * ratio
        
        # 투영 변환 수행
        cv_img = np.array(raw_img)
        warped = four_point_transform(cv_img, original_pts)
        final_img = Image.fromarray(warped)
        
        # 필터 적용
        if source_type == '사진 촬영본':
            final_img = apply_filters(final_img, lighting, surface, flooring_mode, brightness, sharpness)
        
        st.success("✅ 영역 지정 완료! 아래 변환된 이미지를 확인하세요.")
        st.image(final_img, caption="최종 분석 이미지 (쫙 펴짐!)", width=300)

        if st.button("🔍 이 이미지로 검색 시작", type="primary"):
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

    # 6. 결과 출력
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
