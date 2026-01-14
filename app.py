import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import requests
import cv2
import base64
from PIL import Image, ImageEnhance
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------
# 🚑 [긴급 패치] Streamlit 최신 버전 호환성 해결 코드 (필수)
# 사라진 image_to_url 함수를 강제로 만들어서 주입합니다.
# -----------------------------------------------------------
import streamlit.elements.image as st_image

def local_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="auto", image_id=None):
    """PIL 이미지를 HTML에서 볼 수 있는 Base64 주소로 변환"""
    buffered = BytesIO()
    try:
        fmt = image.format if image.format else "PNG"
    except:
        fmt = "PNG"
    image.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{img_str}"

# 라이브러리가 찾을 수 있도록 함수 주입 (Monkey Patching)
if not hasattr(st_image, 'image_to_url'):
    st_image.image_to_url = local_image_to_url
# -----------------------------------------------------------

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

# --- [2] 투영 변환 (Perspective Transform) 로직 ---
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

# --- [3] 통합 이미지 보정 함수 (모든 옵션 포함) ---
def apply_filters(img, source_type, lighting, surface, flooring_mode, brightness, sharpness):
    if source_type == '이미지 파일 (스캔/디지털)':
        return img # 원본은 보정 패스

    # 1. 조명 보정
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split()
        b = b.point(lambda i: i * 1.2)
        img = Image.merge('RGB', (r, g, b))
    elif lighting == '형광등 (푸른/녹색 조명)':
        r, g, b = img.split()
        r = r.point(lambda i: i * 1.1)
        img = Image.merge('RGB', (r, g, b))
    
    # 2. 표면/재질/마루 특화 보정
    enhancer_con = ImageEnhance.Contrast(img)
    enhancer_shp = ImageEnhance.Sharpness(img)

    if flooring_mode != '해당 없음':
        # [마루 특화] 선명도 대폭 강화 (패턴 인식률 향상)
        img = enhancer_shp.enhance(2.0)
        img = enhancer_con.enhance(1.1)
    else:
        # [일반 자재] 표면 질감 반영
        if surface == '하이그로시 (반사 심함)':
            img = enhancer_con.enhance(1.5) # 대비 강화
        elif surface == '매트/엠보 (무광)':
            img = enhancer_con.enhance(1.2) # 대비 약간 강화
            
        if sharpness != 1.0:
            img = enhancer_shp.enhance(sharpness)
        
    # 3. 밝기 보정 (슬라이더)
    if brightness != 1.0:
        enhancer_bri = ImageEnhance.Brightness(img)
        img = enhancer_bri.enhance(brightness)
        
    return img

# --- [4] 메인 UI ---
st.set_page_config(layout="wide", page_title="스마트 자재 검색")
st.title("🏭 스마트 자재 패턴 검색 (풀옵션)")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

uploaded = st.file_uploader("자재 이미지를 업로드하세요", type=['jpg', 'png', 'tif', 'jpeg'])

if uploaded:
    st.markdown("### 🛠️ 이미지 전처리 및 영역 지정")
    
    # [옵션 부활] 조명, 재질, 마루, 밝기, 선명도, 회전 모두 포함
    with st.expander("📸 촬영 환경 및 고급 설정 (클릭하여 열기)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            source_type = st.radio("원본 종류", ['사진 촬영본', '이미지 파일 (스캔/디지털)'])
        with c2:
            lighting = st.selectbox("조명 색상", ['일반/자연광', '백열등 (누런 조명)', '형광등 (푸른/녹색 조명)'], disabled=(source_type!='사진 촬영본'))
        with c3:
            surface = st.selectbox("표면 재질", ['일반', '하이그로시 (반사 심함)', '매트/엠보 (무광)'], disabled=(source_type!='사진 촬영본'))
        with c4:
            flooring_mode = st.selectbox("마루/바닥재 모드", ['해당 없음', '일반 마루', '헤링본/쉐브론'], disabled=(source_type!='사진 촬영본'))

        c5, c6, c7 = st.columns(3)
        with c5:
            # 회전은 캔버스에 넣기 전에 PIL 단계에서 처리
            rotation = st.radio("사진 회전", [0, 90, 180, 270], horizontal=True, format_func=lambda x: f"↩️ {x}도" if x else "원본")
        with c6:
            brightness = st.slider("💡 밝기", 0.5, 2.0, 1.0, 0.1) if source_type == '사진 촬영본' else 1.0
        with c7:
            sharpness = st.slider("🔪 선명도", 0.0, 3.0, 1.5, 0.1) if source_type == '사진 촬영본' else 1.0

    # 1. 이미지 로드 및 기본 회전 적용
    original_image = Image.open(uploaded).convert('RGB')
    if rotation != 0:
        original_image = original_image.rotate(-rotation, expand=True)

    # 2. 캔버스용 리사이징
    canvas_width = 600
    w_percent = (canvas_width / float(original_image.size[0]))
    h_size = int((float(original_image.size[1]) * float(w_percent)))
    resized_image = original_image.resize((canvas_width, h_size))
    
    # 팁 출력
    if flooring_mode == '헤링본/쉐브론':
        st.info("💡 **[Tip]** 헤링본은 여러 쪽이 섞여도 좋으니 **넓게** 영역을 잡아주세요.")
    else:
        st.info("👇 **이미지 위에서 [4개 꼭지점]을 마우스로 콕콕 찍으세요.** (자동으로 펴줍니다)")

    # 3. 캔버스 호출 (Monkey Patch 덕분에 PIL 객체 바로 사용 가능!)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=resized_image, # 여기서 에러가 났었지만 이제 해결됨!
        update_streamlit=True,
        height=h_size,
        width=canvas_width,
        drawing_mode="polygon",
        key="canvas",
    )

    pts = []
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            path = objects[-1]["path"]
            for p in path:
                if p[0] == 'L' or p[0] == 'M': pts.append([p[1], p[2]])

    # 4. 점 4개가 찍히면 변환 및 검색 버튼 활성화
    if len(pts) >= 4:
        # 투영 변환 수행
        pts = np.array(pts[:4], dtype="float32")
        ratio = original_image.size[0] / canvas_width
        original_pts = pts * ratio
        cv_img = np.array(original_image)
        warped = four_point_transform(cv_img, original_pts)
        
        # 보정 필터 적용
        final_img = Image.fromarray(warped)
        final_img = apply_filters(final_img, source_type, lighting, surface, flooring_mode, brightness, sharpness)
        
        # 결과 미리보기
        c_res1, c_res2 = st.columns(2)
        with c_res1: st.image(resized_image, caption="선택 영역", width=300)
        with c_res2: st.image(final_img, caption="최종 분석 이미지 (보정됨)", width=300)

        if st.button("🔍 이 조건으로 검색 시작", type="primary"):
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

    # 5. 결과 출력
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
