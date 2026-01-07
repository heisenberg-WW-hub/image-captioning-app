"""
=============================================================================
utils.py - Module quản lý các mô hình Image Captioning
=============================================================================

File này chứa class ImageCaptioningModels để quản lý việc tải và sử dụng
3 mô hình sinh mô tả ảnh:
    1. ViT-GPT2: Vision Transformer + GPT-2 (nlpconnect/vit-gpt2-image-captioning)
    2. BLIP-Large: Salesforce BLIP (Salesforce/blip-image-captioning-large)
    3. GIT: Microsoft Generative Image-to-text (microsoft/git-large-coco)

Các tính năng chính:
    - Tải mô hình theo yêu cầu (lazy loading) để tiết kiệm bộ nhớ
    - Hỗ trợ GPU (CUDA) nếu có sẵn
    - Sinh mô tả với các tham số có thể điều chỉnh (temperature, top_k, top_p, ...)
    - Giải phóng bộ nhớ khi không cần thiết

Tác giả: Đồ án 2 - 2024
=============================================================================
"""

# =============================================================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# =============================================================================

import torch  # Thư viện deep learning chính
from PIL import Image  # Xử lý hình ảnh

# Import các class từ thư viện Transformers của Hugging Face
from transformers import (
    # Cho mô hình ViT-GPT2
    VisionEncoderDecoderModel,  # Kiến trúc encoder-decoder kết hợp vision và text
    ViTImageProcessor,  # Tiền xử lý ảnh cho Vision Transformer
    AutoTokenizer,  # Tokenizer tự động cho text
    
    # Cho mô hình BLIP
    BlipProcessor,  # Tiền xử lý cho BLIP (cả ảnh và text)
    BlipForConditionalGeneration,  # Mô hình BLIP cho sinh caption
    
    # Cho mô hình GIT
    AutoProcessor,  # Processor tự động
    AutoModelForCausalLM  # Mô hình ngôn ngữ tự động hồi quy
)

import warnings
warnings.filterwarnings('ignore')  # Tắt các cảnh báo không cần thiết


# =============================================================================
# CLASS IMAGEСAPTIONINGMODELS - QUẢN LÝ CÁC MÔ HÌNH SINH MÔ TẢ ẢNH
# =============================================================================

class ImageCaptioningModels:
    """
    Class quản lý việc tải và sử dụng các mô hình sinh mô tả ảnh.
    
    Attributes:
        models (dict): Dictionary lưu trữ các mô hình đã tải
        processors (dict): Dictionary lưu trữ các processor/tokenizer tương ứng
        device (torch.device): Thiết bị chạy mô hình (CPU hoặc GPU)
    
    Ví dụ sử dụng:
        >>> manager = ImageCaptioningModels()
        >>> manager.load_vit_gpt2()  # Tải mô hình ViT-GPT2
        >>> caption = manager.predict_vit_gpt2(image)  # Sinh caption
    """
    
    def __init__(self):
        """
        Khởi tạo ImageCaptioningModels.
        
        - Tạo dictionary rỗng để lưu models và processors
        - Tự động phát hiện và sử dụng GPU nếu có (CUDA)
        - In ra thiết bị đang sử dụng
        """
        self.models = {}  # Lưu trữ các mô hình đã tải {tên: model}
        self.processors = {}  # Lưu trữ processor tương ứng {tên: processor}
        
        # Kiểm tra GPU có sẵn không, nếu có thì dùng GPU, không thì dùng CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    # =========================================================================
    # PHẦN 1: CÁC HÀM TẢI MÔ HÌNH (LOAD MODELS)
    # =========================================================================
    
    def load_vit_gpt2(self):
        """
        Tải mô hình ViT-GPT2 từ Hugging Face Hub.
        
        Mô hình ViT-GPT2 kết hợp:
        - Vision Transformer (ViT): Mã hóa ảnh thành vector đặc trưng
        - GPT-2: Sinh văn bản mô tả từ vector đặc trưng
        
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        
        Lưu ý:
            - Chỉ tải nếu chưa tải trước đó (kiểm tra trong self.models)
            - Mô hình được chuyển sang GPU nếu có sẵn
        """
        try:
            # Kiểm tra xem model đã được tải chưa để tránh tải lại
            if 'vit_gpt2' not in self.models:
                model_name = "nlpconnect/vit-gpt2-image-captioning"
                
                # Tải mô hình VisionEncoderDecoder và chuyển sang device (GPU/CPU)
                model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
                
                # Tải feature extractor (tiền xử lý ảnh) và tokenizer (xử lý text)
                feature_extractor = ViTImageProcessor.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Lưu vào dictionary để sử dụng sau
                self.models['vit_gpt2'] = model
                self.processors['vit_gpt2'] = (feature_extractor, tokenizer)  # Tuple gồm 2 thành phần
                print("ViT-GPT2 model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading ViT-GPT2: {e}")
            return False
    
    def load_blip_large(self):
        """
        Tải mô hình BLIP-Large từ Salesforce.
        
        BLIP (Bootstrapping Language-Image Pre-training) là mô hình
        multimodal được huấn luyện trên dữ liệu lớn với kỹ thuật
        bootstrapping để cải thiện chất lượng caption.
        
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        """
        try:
            if 'blip_large' not in self.models:
                model_name = "Salesforce/blip-image-captioning-large"
                
                # BLIP chỉ cần 1 processor (xử lý cả ảnh và text)
                processor = BlipProcessor.from_pretrained(model_name)
                model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
                
                self.models['blip_large'] = model
                self.processors['blip_large'] = processor
                print("BLIP-Large model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading BLIP-Large: {e}")
            return False
    
    def load_git(self):
        """
        Tải mô hình Microsoft GIT (Generative Image-to-text Transformer).
        
        GIT là mô hình của Microsoft với kiến trúc đơn giản nhưng hiệu quả,
        được huấn luyện trên tập COCO dataset.
        
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        """
        try:
            if 'git' not in self.models:
                model_name = "microsoft/git-large-coco"
                
                # Sử dụng AutoProcessor và AutoModelForCausalLM
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                
                self.models['git'] = model
                self.processors['git'] = processor
                print("Microsoft GIT model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Microsoft GIT: {e}")
            return False
    
    # =========================================================================
    # PHẦN 2: CÁC HÀM SINH MÔ TẢ (PREDICTION FUNCTIONS)
    # =========================================================================
    
    def predict_vit_gpt2(self, image, max_length=50, num_beams=4, temperature=1.0, 
                         top_k=50, top_p=1.0, repetition_penalty=1.0, do_sample=False):
        """
        Sinh mô tả ảnh sử dụng mô hình ViT-GPT2.
        
        Args:
            image (PIL.Image): Ảnh đầu vào cần sinh mô tả
            max_length (int): Độ dài tối đa của caption (số token)
            num_beams (int): Số beam cho beam search (chỉ dùng khi do_sample=False)
            temperature (float): Độ ngẫu nhiên (cao = sáng tạo hơn)
            top_k (int): Số từ được xem xét ở mỗi bước sinh
            top_p (float): Ngưỡng nucleus sampling (0.0-1.0)
            repetition_penalty (float): Hệ số phạt khi lặp từ (>1.0 = phạt mạnh hơn)
            do_sample (bool): True = sampling ngẫu nhiên, False = beam search
        
        Returns:
            str: Mô tả được sinh ra (đã capitalize)
        
        Giải thích các tham số:
            - temperature: Điều khiển độ "sáng tạo". 
              Thấp (0.1-0.5) = chắc chắn, cao (1.0-2.0) = đa dạng
            - top_k: Chỉ xét K từ có xác suất cao nhất
            - top_p: Chỉ xét các từ cho đến khi tổng xác suất đạt p
            - num_beams: Số nhánh trong beam search (cao = chất lượng tốt hơn, chậm hơn)
        """
        # Kiểm tra model đã tải chưa
        if 'vit_gpt2' not in self.models:
            return "Model not loaded"
        
        model = self.models['vit_gpt2']
        feature_extractor, tokenizer = self.processors['vit_gpt2']
        
        # Chuyển ảnh sang RGB nếu cần (một số ảnh có thể là RGBA hoặc grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Tiền xử lý ảnh: resize, normalize, chuyển thành tensor
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        # Sinh caption với gradient tắt (inference mode)
        with torch.no_grad():
            # Cấu hình các tham số sinh văn bản
            gen_kwargs = {
                "max_length": max_length,
                "early_stopping": True,  # Dừng sớm khi gặp token kết thúc
                "repetition_penalty": repetition_penalty,
            }
            
            # Nếu dùng sampling (ngẫu nhiên)
            if do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_k": top_k if top_k > 0 else None,  # 0 = không giới hạn
                    "top_p": top_p,
                    "num_beams": 1,  # Sampling không dùng beam search
                })
            # Nếu dùng beam search (deterministic)
            else:
                gen_kwargs.update({
                    "do_sample": False,
                    "num_beams": num_beams,
                })
            
            # Gọi hàm generate để sinh caption
            output_ids = model.generate(pixel_values, **gen_kwargs)
        
        # Giải mã token IDs thành văn bản, bỏ các token đặc biệt
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.capitalize()  # Viết hoa chữ cái đầu
    
    def predict_blip_large(self, image, max_length=50, num_beams=5, temperature=0.7,
                           top_k=50, top_p=0.9, repetition_penalty=1.0, do_sample=True):
        """
        Sinh mô tả ảnh sử dụng mô hình BLIP-Large.
        
        Tương tự predict_vit_gpt2 nhưng sử dụng kiến trúc BLIP.
        BLIP thường cho kết quả chính xác hơn nhờ được huấn luyện
        trên dữ liệu chất lượng cao.
        
        Args:
            (Tương tự predict_vit_gpt2)
        
        Returns:
            str: Mô tả được sinh ra
        """
        if 'blip_large' not in self.models:
            return "Model not loaded"
        
        model = self.models['blip_large']
        processor = self.processors['blip_large']
        
        # Chuyển sang RGB nếu cần
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Tiền xử lý ảnh với BLIP processor
        inputs = processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            gen_kwargs = {
                "max_length": max_length,
                "repetition_penalty": repetition_penalty,
            }
            
            if do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_k": top_k if top_k > 0 else None,
                    "top_p": top_p,
                    "num_beams": 1,
                })
            else:
                gen_kwargs.update({
                    "do_sample": False,
                    "num_beams": num_beams,
                })
            
            # BLIP sử dụng **inputs thay vì pixel_values riêng
            output_ids = model.generate(**inputs, **gen_kwargs)
        
        # Giải mã kết quả
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.capitalize()
    
    def predict_git(self, image, max_length=50, num_beams=5, temperature=0.7,
                    top_k=50, top_p=0.9, repetition_penalty=1.0, do_sample=True):
        """
        Sinh mô tả ảnh sử dụng mô hình Microsoft GIT.
        
        GIT (Generative Image-to-text Transformer) có kiến trúc
        đơn giản với một transformer duy nhất xử lý cả ảnh và text.
        
        Args:
            (Tương tự predict_vit_gpt2)
        
        Returns:
            str: Mô tả được sinh ra
        """
        if 'git' not in self.models:
            return "Model not loaded"
        
        model = self.models['git']
        processor = self.processors['git']
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Tiền xử lý ảnh
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            gen_kwargs = {
                "max_length": max_length,
                "repetition_penalty": repetition_penalty,
            }
            
            if do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_k": top_k if top_k > 0 else None,
                    "top_p": top_p,
                    "num_beams": 1,
                })
            else:
                gen_kwargs.update({
                    "do_sample": False,
                    "num_beams": num_beams,
                })
            
            # GIT sử dụng pixel_values làm input
            output_ids = model.generate(pixel_values=inputs.pixel_values, **gen_kwargs)
        
        # GIT sử dụng batch_decode và lấy phần tử đầu tiên
        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption.capitalize()
    
    # =========================================================================
    # PHẦN 3: CÁC HÀM TIỆN ÍCH (UTILITY FUNCTIONS)
    # =========================================================================
    
    def predict(self, model_name, image, **kwargs):
        """
        Hàm dự đoán thống nhất - gọi model tương ứng dựa trên tên.
        
        Giúp đơn giản hóa việc gọi các model khác nhau với cùng interface.
        
        Args:
            model_name (str): Tên model ("ViT-GPT2", "BLIP-Large", hoặc "GIT")
            image (PIL.Image): Ảnh đầu vào
            **kwargs: Các tham số sinh caption khác
        
        Returns:
            str: Mô tả được sinh ra
        
        Ví dụ:
            >>> caption = manager.predict("BLIP-Large", image, max_length=60)
        """
        if model_name == "ViT-GPT2":
            return self.predict_vit_gpt2(image, **kwargs)
        elif model_name == "BLIP-Large":
            return self.predict_blip_large(image, **kwargs)
        elif model_name == "GIT":
            return self.predict_git(image, **kwargs)
        else:
            return f"Model {model_name} not supported"
    
    def unload_model(self, model_name):
        """
        Giải phóng một model khỏi bộ nhớ.
        
        Hữu ích khi cần tiết kiệm RAM/VRAM, đặc biệt trên các máy
        có tài nguyên hạn chế.
        
        Args:
            model_name (str): Tên model cần giải phóng
        
        Returns:
            bool: True nếu giải phóng thành công, False nếu model không tồn tại
        
        Ví dụ:
            >>> manager.unload_model("ViT-GPT2")  # Giải phóng ViT-GPT2
        """
        # Chuyển tên model về dạng key (viết thường, thay - bằng _)
        model_key = model_name.lower().replace("-", "_")
        
        if model_key in self.models:
            # Xóa model và processor khỏi dictionary
            del self.models[model_key]
            del self.processors[model_key]
            
            # Giải phóng bộ nhớ GPU nếu đang dùng
            torch.cuda.empty_cache()
            
            print(f"{model_name} unloaded")
            return True
        return False