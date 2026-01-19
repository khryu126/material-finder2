import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import ssl
import cv2
import requests
import base64
import gc
import time
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as k_image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# [0] 환경 설정
ssl._create_default_https_context = ssl._create_unverified_context

# --- [1] 유틸리티 및 데이터 로드 ---
def get_direct_url(url):
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: return url
    if 'file/d/' in url: file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url: file_id = url.split('id=')[1].split('&')[0]
    else: return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

@st.cache_data(ttl=3600)
def get_image_as_base64(url):
    try:
        # TIF 지원 패치 유지: Pillow로 열어서 PNG로 변환 후 전송
        r = requests.get(get_direct_url(url), timeout=15)
        img = Image.open(BytesIO(r.content))
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return None

def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    return pd.DataFrame()

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

@st.cache_resource
def init_resources():
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    with open('material_features.pkl', 'rb') as f:
        full_feature_db = pickle.load(f)
    feature_db = {k: v[:2048] for k, v in full_feature_db.items()}
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    agg_stock, stock_date = {}, "확인불가"
    if not df_stock.empty:
        df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
        agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
        
        # [신규 날짜 로직] 정산일자가 문자열(YYYY-MM-DD)이거나 비어있어도 첫 줄에서 날짜 추출
        if '정산일자' in df_stock.columns:
            valid_dates = df_stock['정산일자'].dropna()
            if not valid_dates.empty:
                raw_val = valid_dates.iloc[0] # 데이터 파일의 날짜가 적힌 첫 번째 행 사용
                try:
                    # 숫자로만 된 날짜라면 정수형 변환 (예: 20241031.0 -> 20241031)
                    stock_date = str(int(float(raw_val)))
                except:
                    # 문자열 형태라면 그대로 사용 (예: 2026-01-19)
                    stock_date = str(raw_val)
                    
    return model_res, feature_db, df_path, df_info, agg_stock, stock_date

res_model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# --- [2] 이미지 처리 엔진 (동일) ---
def apply_advanced_correction(img, state):
    img = ImageEnhance.Brightness(img).enhance(state['bri'])
    img = ImageEnhance.Contrast(img).enhance(state['con'])
    img = ImageEnhance.Sharpness(img).enhance(state['shp'])
    img = ImageEnhance.Color(img).enhance(state['sat'])
    img_np = np.array(img).astype(np.float32)
    img_np *= state['exp']
    temp = state['temp']
    if temp > 1.0: img_np[:, :, 0] *= temp; img_np[:, :, 2] /= temp
    elif temp < 1.0: img_np[:, :, 2] *= (2.0-temp); img_np[:, :, 0] /= (2.0-temp)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    hue = state['hue']
    if hue != 0:
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        img = Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return img

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    w = max(int(np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))), int(np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))))
    h = max(int(np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))), int(np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

# --- [3] Deco Finder Light UI ---
st.set_page_config(layout="wide", page_title="Deco Finder Light")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #B67741; color: white; border-radius: 4px; border: none; font-weight: bold; }
    .stExpander { border: 1px solid #B67741; border-radius: 5px; background-color: white; }
    h1 { color: #B67741; font-family: 'Arial Black', sans-serif; }
    .stock-tag { font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 0.9rem; margin-top: 5px; display: inline-block; }
    .name-tag { font-size: 0.85rem; color: #666; margin-top: 2px; height: 1.2rem; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

st.title("Deco Finder Light")

if 'adj_state' not in st.session_state:
    st.session_state['adj_state'] = {'bri': 1.0, 'con': 1.0, 'shp': 1.0, 'sat': 1.0, 'exp': 1.0, 'temp': 1.0, 'hue': 0}
if 'last_mat' not in st.session_state: st.session_state['last_mat'] = "일반"
if 'res_all' not in st.session_state: st.session_state.update({'res_all': [], 'res_stock': [], 'points': [], 'search_done': False, 'refresh_count': 0})

if st.sidebar.button("🔄 전체 초기화", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.session_state['adj_state'] = {'bri': 1.0, 'con': 1.0, 'shp': 1.0, 'sat': 1.0, 'exp': 1.0, 'temp': 1.0, 'hue': 0}
    st.session_state.update({'res_all': [], 'res_stock': [], 'points': [], 'search_done': False, 'refresh_count': 0, 'last_mat': '일반'})
    gc.collect(); st.rerun()

st.sidebar.markdown(f"📦 **재고 정산일:** \n{stock_date}")

uploaded = st.file_uploader("📸 자재 사진 업로드", type=['jpg','png','jpeg'])

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state.update({'points': [], 'search_done': False, 'res_all': [], 'res_stock': [], 'current_img_name': uploaded.name, 'proc_img': Image.open(uploaded).convert('RGB')})
        gc.collect(); st.rerun()

    working_img = st.session_state['proc_img']
    w, h = working_img.size

    with st.expander("🛠️ 고급 이미지 보정", expanded=False):
        def adj_btn(label, key, step):
            c_l, c_v, c_m, c_p = st.columns([2, 1, 1, 1])
            c_l.markdown(f"**{label}**")
            c_v.text(f"{st.session_state['adj_state'][key]:.1f}")
            if c_m.button("➖", key=f"dec_{key}"): st.session_state['adj_state'][key] = round(max(0.1, st.session_state['adj_state'][key] - step), 2); st.rerun()
            if c_p.button("➕", key=f"inc_{key}"): st.session_state['adj_state'][key] = round(st.session_state['adj_state'][key] + step, 2); st.rerun()
        adj_btn("밝기", "bri", 0.1); adj_btn("대비", "con", 0.1); adj_btn("선명도", "shp", 0.5)
        adj_btn("채도", "sat", 0.1); adj_btn("노출", "exp", 0.1); adj_btn("온도", "temp", 0.1)

    scale = st.radio("🔍 보기 크기:", [0.1, 0.3, 0.5, 0.7, 1.0], index=2, horizontal=True)
    
    col_ui, col_pad = st.columns([1, 2])
    with col_ui:
        source_type = st.radio("출처", ['📸 촬영', '💻 디지털'], horizontal=True)
        mat_type = st.selectbox("분류 (선택 시 자동 보정)", ['일반', '우드', '유광', '패브릭', '석재'])
        
        if mat_type != st.session_state['last_mat']:
            st.session_state['last_mat'] = mat_type
            if mat_type == '우드': st.session_state['adj_state'].update({'con': 1.2, 'shp': 1.5})
            elif mat_type == '유광': st.session_state['adj_state'].update({'con': 1.1, 'exp': 0.8})
            elif mat_type == '석재': st.session_state['adj_state'].update({'shp': 2.0})
            elif mat_type == '패브릭': st.session_state['adj_state'].update({'con': 1.3})
            else: st.session_state['adj_state'] = {'bri': 1.0, 'con': 1.0, 'shp': 1.0, 'sat': 1.0, 'exp': 1.0, 'temp': 1.0, 'hue': 0}
            st.rerun()
            
        s_mode = st.radio("분석 모드", ["종합(컬러+패턴)", "패턴 중심(흑백)"], horizontal=True)
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔄 새로고침", use_container_width=True): st.session_state['refresh_count'] = time.time(); st.rerun()
        with c_btn2:
            if st.button("↪️ 90도 회전", use_container_width=True):
                st.session_state['proc_img'] = working_img.transpose(Image.ROTATE_270)
                st.session_state['points'] = []; st.rerun()
        
        if st.button("⏹️ 전체 선택 (Select All)", use_container_width=True, type="secondary"):
            st.session_state['points'] = [(0, 0), (w, 0), (w, h), (0, h)]; st.rerun()
        
        if st.button("📍 점 다시찍기", use_container_width=True): st.session_state['points'] = []; st.rerun()

    with col_pad:
        d_img = working_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(d_img)
        for i, p in enumerate(st.session_state['points']):
            px, py = p[0]*scale, p[1]*scale
            draw.ellipse((px-8, py-8, px+8, py+8), fill='#B67741', outline='white', width=2)
            draw.text((px+12, py-12), str(i+1), fill='red')
        if len(st.session_state['points']) == 4:
            rect_pts = np.array(st.session_state['points'], dtype="float32")
            draw.polygon([tuple((p[0]*scale, p[1]*scale)) for p in rect_pts], outline='#00FF00', width=3)
        coords = streamlit_image_coordinates(d_img, key=f"deco_{st.session_state['refresh_count']}")
        if coords and len(st.session_state['points']) < 4:
            new_p = (coords['x']/scale, coords['y']/scale)
            if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                st.session_state['points'].append(new_p); st.rerun()

    if len(st.session_state['points']) == 4:
        warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
        final_img = Image.fromarray(warped)
        final_img = apply_advanced_correction(final_img, st.session_state['adj_state'])
        if "흑백" in s_mode: final_img = final_img.convert("L").convert("RGB")
        st.image(final_img, width=300, caption="분석 대상")
        if st.button("🔍 Deco Finder Light 검색 시작", type="primary", use_container_width=True):
            with st.spinner('ResNet 연산 중...'):
                x_res = k_image.img_to_array(final_img.resize((224, 224)))
                q_res = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
                results = []
                for fn, db_vec in feature_db.items():
                    score = cosine_similarity([q_res], [db_vec])[0][0]
                    d_key = get_digits(fn); match_info = df_info[df_info['상품코드'].apply(get_digits) == d_key]
                    f_code = match_info.iloc[0]['상품코드'] if not match_info.empty else fn.split('.')[0]
                    p_name = match_info.iloc[0]['상품명'] if not match_info.empty else ""
                    f_key = str(f_code).strip().upper(); qty = agg_stock.get(f_key, 0)
                    url_row = df_path[df_path['추출된_품번'].apply(get_digits) == d_key]
                    url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                    if url: results.append({'formal': f_code, 'name': p_name, 'score': score, 'url': url, 'stock': qty})
                results.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['res_all'] = results[:15]
                st.session_state['res_stock'] = [r for r in results if r['stock'] > 0][:15]
                gc.collect(); st.session_state['search_done'] = True; st.rerun()

# --- [4] 결과 출력 ---
if st.session_state.get('search_done') and st.session_state.get('res_all'):
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 전체 결과", "✅ 재고 보유"])
    def display_grid(items):
        if not items: st.warning("결과 없음"); return
        for row_idx in range(0, len(items), 5):
            cols = st.columns(5)
            for col_idx in range(5):
                idx = row_idx + col_idx
                if idx < len(items):
                    item = items[idx]
                    with cols[col_idx]:
                        st.markdown(f"**{idx+1}위: {item['formal']}**")
                        st.markdown(f"<div class='name-tag'>{item['name']}</div>", unsafe_allow_html=True)
                        if item['stock'] >= 100: st.markdown(f"<span class='stock-tag' style='color:#155724; background-color:#d4edda;'>보유: {item['stock']:,}m</span>", unsafe_allow_html=True)
                        elif item['stock'] > 0: st.markdown(f"<span class='stock-tag' style='color:#856404; background-color:#fff3cd;'>보유: {item['stock']:,}m</span>", unsafe_allow_html=True)
                        else: st.markdown(f"<span class='stock-tag' style='color:#721c24; background-color:#f8d7da;'>재고 없음</span>", unsafe_allow_html=True)
                        st.caption(f"유사도: {item['score']:.1%}")
                        with st.expander("🖼️ 상세보기", expanded=False):
                            b64 = get_image_as_base64(item['url'])
                            if b64: st.image(b64, use_container_width=True)
                            st.write(f"**품명:** {item['name']}")
    with tab1: display_grid(st.session_state['res_all'])
    with tab2: display_grid(st.session_state['res_stock'])
