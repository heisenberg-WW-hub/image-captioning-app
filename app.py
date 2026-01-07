"""
=============================================================================
app.py - Ứng dụng Streamlit sinh mô tả ảnh (Image Captioning)
=============================================================================

File này là điểm khởi chạy chính của ứng dụng web Streamlit.
Cho phép người dùng:
    1. Tải ảnh lên (từ file, URL, hoặc webcam)
    2. Chọn mô hình AI để sinh mô tả
    3. Điều chỉnh các tham số sinh caption
    4. So sánh kết quả giữa các mô hình
    5. Dịch mô tả sang tiếng Việt

Cách chạy:
    streamlit run app.py

Tác giả: Đồ án 2 - 2024
=============================================================================
"""

# =============================================================================
# IMPORT CÁC THƯ VIỆN
# =============================================================================

import streamlit as st  # Framework web app
import requests  # Gọi HTTP để tải ảnh từ URL
from PIL import Image  # Xử lý hình ảnh
import io  # Xử lý byte stream
import time  # Đo thời gian xử lý
import torch  # Deep learning framework
from utils import ImageCaptioningModels  # Class quản lý các model (từ file utils.py)
from deep_translator import GoogleTranslator  # Dịch văn bản sang tiếng Việt


# =============================================================================
# CẤU HÌNH TRANG WEB
# =============================================================================

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống sinh mô tả ảnh",  # Tiêu đề tab trình duyệt
    page_icon="🖼️",  # Icon tab
    layout="wide"  # Sử dụng toàn bộ chiều rộng màn hình
)


# =============================================================================
# PRESET CONFIGURATIONS - CÁC CẤU HÌNH SẴN CÓ
# =============================================================================

# Dictionary chứa các bộ tham số được cấu hình sẵn
# Người dùng có thể chọn nhanh thay vì phải điều chỉnh từng tham số
PRESETS = {
    # Preset 1: Sáng tạo - Tạo mô tả đa dạng, độc đáo
    "🎨 Sáng tạo (Creative)": {
        "max_length": 60,  # Cho phép mô tả dài hơn
        "num_beams": 4,
        "temperature": 1.2,  # Cao = ngẫu nhiên hơn
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,  # Phạt lặp từ
        "do_sample": True,  # Bật sampling
        "description": "Tạo mô tả đa dạng, sáng tạo hơn"
    },
    
    # Preset 2: Cân bằng - Mặc định, phù hợp đa số trường hợp
    "⚖️ Cân bằng (Balanced)": {
        "max_length": 50,
        "num_beams": 4,
        "temperature": 0.7,  # Trung bình
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "description": "Cân bằng giữa sáng tạo và chính xác"
    },
    
    # Preset 3: Chính xác - Kết quả ổn định, không ngẫu nhiên
    "🎯 Chính xác (Precise)": {
        "max_length": 40,
        "num_beams": 5,  # Nhiều beam = kết quả tốt hơn
        "temperature": 0.3,  # Thấp = ít ngẫu nhiên
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "do_sample": False,  # Tắt sampling, dùng beam search
        "description": "Mô tả chính xác, nhất quán, dùng beam search"
    },
    
    # Preset 4: Tùy chỉnh - Cho phép người dùng điều chỉnh tất cả
    "🔧 Tùy chỉnh (Custom)": {
        "max_length": 50,
        "num_beams": 4,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "description": "Tự điều chỉnh tất cả tham số"
    }
}


# =============================================================================
# KHỞI TẠO VÀ CACHE RESOURCES
# =============================================================================

@st.cache_resource  # Cache model manager để không phải tạo lại mỗi lần refresh
def load_model_manager():
    """
    Tạo và cache instance của ImageCaptioningModels.
    
    Sử dụng @st.cache_resource để:
    - Chỉ tạo 1 lần duy nhất
    - Giữ trong bộ nhớ xuyên suốt session
    - Tiết kiệm thời gian tải lại
    """
    return ImageCaptioningModels()


@st.cache_data  # Cache dữ liệu ảnh từ URL
def load_image_from_url(url):
    """
    Tải và cache hình ảnh từ URL.
    
    Args:
        url (str): Đường dẫn URL của ảnh
    
    Returns:
        PIL.Image: Đối tượng ảnh đã tải
    """
    response = requests.get(url, stream=True, timeout=10)
    return Image.open(io.BytesIO(response.content))


@st.cache_data  # Cache ảnh từ file upload
def load_image_from_file(uploaded_file):
    """
    Tải và cache hình ảnh từ file upload.
    
    Args:
        uploaded_file: File được upload qua Streamlit
    
    Returns:
        PIL.Image: Đối tượng ảnh
    """
    return Image.open(uploaded_file)


# =============================================================================
# GIAO DIỆN CHÍNH - TIÊU ĐỀ
# =============================================================================

# Tiêu đề và mô tả ứng dụng
st.title("🖼️ Sinh mô tả ảnh với nhiều mô hình")
st.markdown("Tạo mô tả tự động cho hình ảnh sử dụng các mô hình AI khác nhau")


# =============================================================================
# SIDEBAR - THANH ĐIỀU KHIỂN BÊN TRÁI
# =============================================================================

with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    # -------------------------------------------------------------------------
    # PHẦN 1: LỰA CHỌN MÔ HÌNH
    # -------------------------------------------------------------------------
    st.subheader("🤖 Lựa chọn mô hình")
    
    # Danh sách các model có sẵn
    model_options = ["ViT-GPT2", "BLIP-Large", "GIT"]
    
    # Dropdown để chọn model
    selected_model = st.selectbox(
        "Chọn mô hình:",
        model_options,
        help="• **ViT-GPT2**: Nhanh, nhẹ\n• **BLIP-Large**: Chính xác cao\n• **GIT**: Microsoft Generative Image-to-text"
    )
    
    # Khởi tạo model manager (được cache)
    model_manager = load_model_manager()
    
    # Nút tải model
    load_clicked = st.button(f"📥 Tải mô hình {selected_model}", use_container_width=True)
    if load_clicked:
        with st.spinner(f"Đang tải {selected_model}..."):
            # Gọi hàm load tương ứng với model được chọn
            if selected_model == "ViT-GPT2":
                success = model_manager.load_vit_gpt2()
            elif selected_model == "BLIP-Large":
                success = model_manager.load_blip_large()
            else:  # GIT
                success = model_manager.load_git()
            
            # Hiển thị kết quả
            if success:
                st.success(f"✅ {selected_model} đã sẵn sàng!")
            else:
                st.error(f"❌ Lỗi khi tải {selected_model}")
    
    st.divider()  # Đường kẻ phân cách
    
    # -------------------------------------------------------------------------
    # PHẦN 2: CẤU HÌNH THAM SỐ (PRESETS)
    # -------------------------------------------------------------------------
    st.subheader("🎛️ Cấu hình tham số")
    
    # Dropdown chọn preset
    selected_preset = st.selectbox(
        "Chọn preset:",
        list(PRESETS.keys()),
        index=1,  # Mặc định chọn "Cân bằng"
        help="Chọn cấu hình sẵn hoặc 'Tùy chỉnh' để điều chỉnh thủ công"
    )
    
    # Lấy config của preset được chọn
    preset_config = PRESETS[selected_preset]
    st.caption(f"ℹ️ {preset_config['description']}")  # Hiển thị mô tả preset
    
    # Kiểm tra xem có phải preset "Tùy chỉnh" không
    is_custom = "Tùy chỉnh" in selected_preset
    
    # -------------------------------------------------------------------------
    # PHẦN 3: ĐIỀU CHỈNH THAM SỐ CHI TIẾT
    # -------------------------------------------------------------------------
    with st.expander("📊 Tham số chi tiết", expanded=is_custom):
        
        # Toggle bật/tắt Sampling
        do_sample = st.checkbox(
            "🎲 Sampling (do_sample)",
            value=preset_config["do_sample"] if not is_custom else True,
            disabled=not is_custom,  # Chỉ cho phép thay đổi ở mode Custom
            help="**BẬT**: Tạo văn bản ngẫu nhiên (dùng temperature, top_k, top_p)\n"
                 "**TẮT**: Dùng beam search cho kết quả ổn định hơn"
        )
        
        st.markdown("---")
        st.markdown("**📏 Kích thước đầu ra**")
        
        # Slider độ dài tối đa
        max_length = st.slider(
            "Độ dài tối đa (max_length)",
            min_value=10,
            max_value=100,
            value=preset_config["max_length"] if not is_custom else 50,
            step=5,
            disabled=not is_custom,
            help="Số token tối đa trong mô tả được tạo. Giá trị lớn = mô tả dài hơn."
        )
        
        st.markdown("---")
        
        # Hiển thị các tham số khác nhau tùy thuộc vào do_sample
        if do_sample:
            # === THAM SỐ SAMPLING ===
            st.markdown("**🌡️ Tham số Sampling**")
            
            # Temperature - điều khiển độ ngẫu nhiên
            temperature = st.slider(
                "Nhiệt độ (temperature)",
                min_value=0.1,
                max_value=2.0,
                value=preset_config["temperature"] if not is_custom else 0.7,
                step=0.1,
                disabled=not is_custom,
                help="Điều khiển độ ngẫu nhiên:\n"
                     "• **Thấp (0.1-0.5)**: Kết quả chính xác, lặp lại\n"
                     "• **Trung bình (0.6-1.0)**: Cân bằng\n"
                     "• **Cao (1.1-2.0)**: Sáng tạo, đa dạng hơn"
            )
            
            # Top-K - giới hạn số từ xem xét
            top_k = st.slider(
                "Top-K",
                min_value=0,
                max_value=100,
                value=preset_config["top_k"] if not is_custom else 50,
                step=5,
                disabled=not is_custom,
                help="Giới hạn số từ được xem xét ở mỗi bước:\n"
                     "• **0**: Không giới hạn\n"
                     "• **Thấp (10-30)**: Tập trung vào từ phổ biến\n"
                     "• **Cao (50-100)**: Cho phép từ ít phổ biến hơn"
            )
            
            # Top-P (Nucleus Sampling)
            top_p = st.slider(
                "Top-P (Nucleus Sampling)",
                min_value=0.1,
                max_value=1.0,
                value=preset_config["top_p"] if not is_custom else 0.9,
                step=0.05,
                disabled=not is_custom,
                help="Chọn từ trong xác suất tích lũy:\n"
                     "• **Thấp (0.5-0.7)**: Chỉ từ có xác suất cao\n"
                     "• **Cao (0.9-1.0)**: Bao gồm nhiều từ hơn"
            )
            
            # Khi dùng sampling thì num_beams = 1 (bắt buộc)
            num_beams = 1
        else:
            # === THAM SỐ BEAM SEARCH ===
            st.markdown("**🔦 Tham số Beam Search**")
            
            # Số beam
            num_beams = st.slider(
                "Số beams (num_beams)",
                min_value=1,
                max_value=10,
                value=preset_config["num_beams"] if not is_custom else 4,
                step=1,
                disabled=not is_custom,
                help="Số lượng beam trong beam search:\n"
                     "• **Thấp (1-2)**: Nhanh hơn\n"
                     "• **Cao (4-10)**: Kết quả tốt hơn nhưng chậm hơn"
            )
            
            # Giá trị mặc định cho các tham số sampling khi không dùng
            temperature = 1.0
            top_k = 50
            top_p = 1.0
        
        st.markdown("---")
        st.markdown("**🔁 Kiểm soát lặp lại**")
        
        # Repetition Penalty - phạt khi lặp từ
        repetition_penalty = st.slider(
            "Phạt lặp từ (repetition_penalty)",
            min_value=1.0,
            max_value=2.0,
            value=preset_config["repetition_penalty"] if not is_custom else 1.0,
            step=0.1,
            disabled=not is_custom,
            help="Phạt khi lặp lại từ đã dùng:\n"
                 "• **1.0**: Không phạt\n"
                 "• **1.2-1.5**: Giảm lặp từ\n"
                 "• **>1.5**: Tránh lặp mạnh"
        )
    
    # Nếu không phải Custom mode, lấy giá trị từ preset
    if not is_custom:
        max_length = preset_config["max_length"]
        num_beams = preset_config["num_beams"]
        temperature = preset_config["temperature"]
        top_k = preset_config["top_k"]
        top_p = preset_config["top_p"]
        repetition_penalty = preset_config["repetition_penalty"]
        do_sample = preset_config["do_sample"]
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # PHẦN 4: CẤU HÌNH RIÊNG CHO TỪNG MODEL (NÂNG CAO)
    # -------------------------------------------------------------------------
    with st.expander("🔬 Cấu hình riêng cho từng model", expanded=False):
        st.caption("Ghi đè tham số cho model cụ thể (tùy chọn)")
        
        # Toggle bật/tắt cấu hình riêng
        use_model_specific = st.checkbox("Sử dụng cấu hình riêng", value=False)
        
        if use_model_specific:
            st.markdown(f"**Cấu hình cho {selected_model}:**")
            
            # Cho phép ghi đè max_length và temperature
            model_max_length = st.slider(
                f"Max length ({selected_model})",
                min_value=10,
                max_value=100,
                value=max_length,
                step=5,
                key=f"model_{selected_model}_max_length"
            )
            
            model_temperature = st.slider(
                f"Temperature ({selected_model})",
                min_value=0.1,
                max_value=2.0,
                value=temperature,
                step=0.1,
                key=f"model_{selected_model}_temperature"
            )
            
            # Ghi đè giá trị
            max_length = model_max_length
            temperature = model_temperature
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # PHẦN 5: QUẢN LÝ BỘ NHỚ
    # -------------------------------------------------------------------------
    st.subheader("💾 Quản lý bộ nhớ")
    
    # Nút xóa cache và giải phóng bộ nhớ
    if st.button("🗑️ Xóa cache và giải phóng bộ nhớ", use_container_width=True):
        st.cache_resource.clear()  # Xóa cache Streamlit
        torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
        st.success("Đã xóa cache và giải phóng bộ nhớ!")
    
    # -------------------------------------------------------------------------
    # PHẦN 6: THÔNG TIN MÔ HÌNH
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("""
    ### ℹ️ Thông tin mô hình:
    - **ViT-GPT2**: Vision Transformer + GPT-2, nhanh
    - **BLIP-Large**: Bootstrapping Language-Image Pre-training, chính xác
    - **GIT**: Microsoft Generative Image-to-text Transformer
    """)


# =============================================================================
# PHẦN CHÍNH - TẢI ẢNH LÊN
# =============================================================================

# Tạo 3 tab để chọn nguồn ảnh
tab1, tab2, tab3 = st.tabs(["📤 Tải ảnh lên", "🌐 Từ URL", "📷 Chụp ảnh"])

# Biến lưu ảnh được tải lên
uploaded_image = None

# -------------------------------------------------------------------------
# TAB 1: TẢI ẢNH TỪ FILE
# -------------------------------------------------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Tải lên hình ảnh",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],  # Các định dạng được hỗ trợ
        help="Chọn hình ảnh từ máy tính của bạn"
    )
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Hình ảnh đã tải lên", use_container_width=True)

# -------------------------------------------------------------------------
# TAB 2: TẢI ẢNH TỪ URL
# -------------------------------------------------------------------------
with tab2:
    url = st.text_input(
        "Nhập URL hình ảnh:",
        placeholder="https://example.com/image.jpg"
    )
    
    if url:
        try:
            # Gọi HTTP GET để tải ảnh
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                uploaded_image = Image.open(io.BytesIO(response.content))
                st.image(uploaded_image, caption="Hình ảnh từ URL", use_container_width=True)
            else:
                st.error(f"Không thể tải ảnh. Mã lỗi: {response.status_code}")
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh: {e}")

# -------------------------------------------------------------------------
# TAB 3: CHỤP ẢNH TỪ WEBCAM
# -------------------------------------------------------------------------
with tab3:
    camera_image = st.camera_input("Chụp ảnh từ webcam")
    if camera_image is not None:
        uploaded_image = Image.open(camera_image)
        st.image(uploaded_image, caption="Ảnh đã chụp", use_container_width=True)


# =============================================================================
# PHẦN TẠO CAPTION - SINH MÔ TẢ ẢNH
# =============================================================================

# Chỉ hiển thị phần này nếu có ảnh được tải lên
if uploaded_image is not None:
    st.divider()
    st.header("🎯 Tạo mô tả hình ảnh")
    
    # Chia layout thành 2 cột
    col1, col2 = st.columns(2)
    
    # -------------------------------------------------------------------------
    # CỘT 1: HIỂN THỊ ẢNH ĐẦU VÀO
    # -------------------------------------------------------------------------
    with col1:
        st.subheader("Hình ảnh đầu vào")
        st.image(uploaded_image, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # CỘT 2: SINH VÀ HIỂN THỊ MÔ TẢ
    # -------------------------------------------------------------------------
    with col2:
        st.subheader("Mô tả được tạo")
        
        # Kiểm tra xem model đã được tải chưa
        model_key = selected_model.lower().replace("-", "_")
        model_loaded = model_key in model_manager.models
        
        if not model_loaded:
            st.warning(f"⚠️ Mô hình {selected_model} chưa được tải. Vui lòng nhấn 'Tải mô hình' trong sidebar.")
        else:
            st.info(f"**Mô hình đang sử dụng:** {selected_model}")
            
            # Hiển thị các tham số đang sử dụng (có thể mở rộng)
            with st.expander("📊 Tham số đang sử dụng"):
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    st.write(f"• **Max length:** {max_length}")
                    st.write(f"• **Temperature:** {temperature}")
                    st.write(f"• **Top-K:** {top_k}")
                with param_col2:
                    st.write(f"• **Top-P:** {top_p}")
                    st.write(f"• **Num beams:** {num_beams}")
                    st.write(f"• **Do sample:** {do_sample}")
                    st.write(f"• **Repetition penalty:** {repetition_penalty}")
            
            # -----------------------------------------------------------------
            # NÚT TẠO MÔ TẢ
            # -----------------------------------------------------------------
            if st.button("🚀 Tạo mô tả", type="primary", use_container_width=True):
                with st.spinner("Đang tạo mô tả..."):
                    start_time = time.time()  # Bắt đầu đo thời gian
                    
                    # Chuẩn bị tham số chung cho tất cả model
                    gen_params = {
                        "max_length": max_length,
                        "num_beams": num_beams,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "do_sample": do_sample
                    }
                    
                    # Gọi model tương ứng để sinh caption
                    if selected_model == "ViT-GPT2":
                        caption = model_manager.predict_vit_gpt2(uploaded_image, **gen_params)
                    elif selected_model == "BLIP-Large":
                        caption = model_manager.predict_blip_large(uploaded_image, **gen_params)
                    else:  # GIT
                        caption = model_manager.predict_git(uploaded_image, **gen_params)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time  # Tính thời gian xử lý
                    
                    # Hiển thị kết quả
                    st.success("✅ Mô tả đã được tạo!")
                    
                    # Lưu caption vào session state để có thể dịch sau
                    st.session_state['current_caption'] = caption
                    st.session_state['caption_translated'] = None
                    
                    # Hiển thị caption trong box
                    st.markdown(f"**📝 Mô tả (English):**")
                    st.info(f"**{caption}**")
                    
                    # Hiển thị thời gian xử lý
                    st.caption(f"⏱️ Thời gian xử lý: {processing_time:.2f} giây")
                    
                    # Nút tải xuống caption dạng file text
                    caption_text = f"Mô tả hình ảnh:\n{caption}\n\nTạo bởi: {selected_model}\nPreset: {selected_preset}"
                    st.download_button(
                        label="📥 Tải mô tả",
                        data=caption_text,
                        file_name="image_caption.txt",
                        mime="text/plain"
                    )
            
            # -----------------------------------------------------------------
            # TÍNH NĂNG DỊCH SANG TIẾNG VIỆT
            # -----------------------------------------------------------------
            # Hiển thị nút dịch nếu đã có caption
            if 'current_caption' in st.session_state and st.session_state['current_caption']:
                st.divider()
                
                if st.button("🇻🇳 Dịch sang tiếng Việt", type="secondary", use_container_width=True):
                    with st.spinner("Đang dịch..."):
                        try:
                            # Sử dụng Google Translator để dịch EN -> VI
                            translator = GoogleTranslator(source='en', target='vi')
                            translated = translator.translate(st.session_state['current_caption'])
                            st.session_state['caption_translated'] = translated
                        except Exception as e:
                            st.error(f"Lỗi khi dịch: {e}")
                
                # Hiển thị bản dịch nếu có
                if st.session_state.get('caption_translated'):
                    st.markdown("**📝 Mô tả (Tiếng Việt):**")
                    st.success(f"**{st.session_state['caption_translated']}**")
    
    # =========================================================================
    # PHẦN SO SÁNH NHIỀU MÔ HÌNH
    # =========================================================================
    st.divider()
    st.header("📊 So sánh nhiều mô hình")
    
    # Nút chạy tất cả model để so sánh
    if st.button("🔍 Chạy tất cả mô hình", type="secondary"):
        models_to_compare = []
        captions = {}  # Lưu caption của từng model
        processing_times = {}  # Lưu thời gian xử lý của từng model
        
        # Chuẩn bị tham số chung
        gen_params = {
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample
        }
        
        # Kiểm tra và tải các model chưa được tải
        for model in model_options:
            model_key = model.lower().replace("-", "_")
            if model_key not in model_manager.models:
                st.info(f"Đang tải {model}...")
                if model == "ViT-GPT2":
                    model_manager.load_vit_gpt2()
                elif model == "BLIP-Large":
                    model_manager.load_blip_large()
                else:
                    model_manager.load_git()
        
        # Tạo caption với từng model và hiển thị progress bar
        progress_bar = st.progress(0)
        for i, model in enumerate(model_options):
            st.write(f"Đang xử lý với {model}...")
            
            start_time = time.time()
            
            # Gọi model tương ứng
            if model == "ViT-GPT2":
                caption = model_manager.predict_vit_gpt2(uploaded_image, **gen_params)
            elif model == "BLIP-Large":
                caption = model_manager.predict_blip_large(uploaded_image, **gen_params)
            else:
                caption = model_manager.predict_git(uploaded_image, **gen_params)
            
            end_time = time.time()
            
            # Lưu kết quả
            captions[model] = caption
            processing_times[model] = end_time - start_time
            
            # Cập nhật progress bar
            progress_bar.progress((i + 1) / len(model_options))
        
        # ---------------------------------------------------------------------
        # HIỂN THỊ KẾT QUẢ SO SÁNH
        # ---------------------------------------------------------------------
        st.subheader("Kết quả so sánh")
        
        # Hiển thị theo cột
        cols = st.columns(len(model_options))
        for idx, (model, col) in enumerate(zip(model_options, cols)):
            with col:
                st.markdown(f"**{model}**")
                st.metric("Thời gian", f"{processing_times[model]:.2f}s")
                st.info(captions[model])
        
        # Bảng tổng hợp
        st.subheader("📋 Tổng hợp")
        comparison_data = {
            "Mô hình": model_options,
            "Mô tả": [captions[m] for m in model_options],
            "Thời gian (s)": [f"{processing_times[m]:.2f}" for m in model_options]
        }
        st.table(comparison_data)


# =============================================================================
# FOOTER - CHÂN TRANG
# =============================================================================

st.divider()
st.markdown("""
---
### 📚 Thông tin thêm:
- **ViT-GPT2**: Sử dụng Vision Transformer để mã hóa ảnh và GPT-2 để tạo văn bản
- **BLIP-Large**: Mô hình đa phương thức được huấn luyện trên tập dữ liệu lớn
- **GIT**: Microsoft Generative Image-to-text Transformer, kiến trúc đơn giản hiệu quả

🔧 **Lưu ý**: Lần đầu chạy sẽ mất thời gian để tải mô hình từ internet.
""")