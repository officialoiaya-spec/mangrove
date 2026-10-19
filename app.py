# Patch untuk cloud environment (wajib di awal)
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import requests
import time

# ==========================================
# KONFIGURASI HALAMAN & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Klasifikasi Mangrove", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8_mangrove_v2_best.pt")

model = load_model()

# ==========================================
# KONFIGURASI LABEL YANG DIIZINKAN
# ==========================================
VALID_MANGROVE_CLASSES = {"avicennia_alba", "rhizophora_apiculata", "sonneratia_alba"}
UNKNOWN_LABEL = "Tidak Dikenali"

# ==========================================
# TAMPILAN UI
# ==========================================
st.title("🌿 Klasifikasi Jenis Mangrove Berdasarkan Daun")
st.write("Aplikasi klasifikasi mangrove (Avicennia Alba, Rhizophora Apiculata, Sonneratia Alba) menggunakan YOLOv11.")

st.sidebar.header("⚙️ Pengaturan")
confidence_threshold = st.sidebar.slider(
    "Tingkat Confidence Min (%) untuk Dikenali", 
    10, 100, 60, 5,
    help="Jika confidence tertinggi < nilai ini, gambar akan ditandai 'Tidak Dikenali'"
)

# Toggle untuk auto-detect area daun
use_leaf_detection = st.sidebar.checkbox(
    "Auto-Deteksi Area Daun (HSV)", 
    value=False,
    help="Jika aktif, sistem akan mendeteksi area berwarna hijau (daun) dan hanya memberi bounding box di area tersebut. Jika nonaktif, bounding box akan mencakup seluruh gambar."
)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Pilih Mode Input:", 
                        ["📷 Upload Gambar", 
                         "🎬 Upload Video", 
                         "🌐 URL Video", 
                         "📹 Webcam Live"])

# ==========================================
# FUNGSI DETEKSI AREA DAUN (HSV)
# ==========================================
def detect_leaf_regions(frame, min_area_ratio=0.01):
    """
    Mendeteksi area daun berwarna hijau menggunakan HSV color thresholding.
    Return: list of (x, y, w, h) bounding boxes
    """
    # Konversi ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Range warna hijau untuk daun (bisa disesuaikan)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Thresholding
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological operations untuk membersihkan noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter berdasarkan area minimum
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * min_area_ratio
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Padding sedikit di sekitar box
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(frame.shape[1] - x, w + 2 * pad)
            h = min(frame.shape[0] - y, h + 2 * pad)
            boxes.append((x, y, w, h))
    
    return boxes

# ==========================================
# FUNGSI KLASIFIKASI
# ==========================================
def classify_region(img):
    """Klasifikasikan gambar dan return (label, confidence, color)"""
    results = model.predict(img, verbose=False)
    probs = results[0].probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item() * 100
    raw_class_name = model.names[top1_idx]
    
    is_valid_mangrove = raw_class_name in VALID_MANGROVE_CLASSES
    
    if is_valid_mangrove and top1_conf >= confidence_threshold:
        final_label = raw_class_name
        color = (0, 200, 0)  # Hijau
    else:
        final_label = UNKNOWN_LABEL
        color = (0, 0, 220)  # Merah
    
    return final_label, top1_conf, color, results

def draw_bounding_box(frame, x, y, w, h, label, conf, color):
    """
    Gambar bounding box dengan label di atasnya.
    Color dalam format RGB.
    """
    # Konversi RGB ke BGR untuk OpenCV
    color_bgr = (color[2], color[1], color[0])
    
    # Gambar rectangle
    thickness = 3
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, thickness)
    
    # Label text
    label_text = f"{label} ({conf:.1f}%)"
    
    # Hitung ukuran text untuk background
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    # Posisi background text (di atas bounding box)
    text_y = y - 10 if y - 10 > text_h else y + text_h + 10
    
    # Background text
    cv2.rectangle(frame, 
                  (x, text_y - text_h - 5), 
                  (x + text_w + 10, text_y + 5), 
                  color_bgr, -1)
    
    # Text
    cv2.putText(frame, label_text, (x + 5, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def process_frame_with_bbox(frame):
    """
    Proses frame: deteksi area daun (atau whole image), klasifikasi, gambar bbox.
    Return: (annotated_frame, list_of_detections)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = []
    
    if use_leaf_detection:
        # Mode auto-detect area daun
        boxes = detect_leaf_regions(frame_rgb)
        
        if len(boxes) == 0:
            # Jika tidak ada area daun terdeteksi, klasifikasi seluruh frame
            label, conf, color, _ = classify_region(frame_rgb)
            h, w = frame.shape[:2]
            frame_rgb = draw_bounding_box(frame_rgb, 10, 10, w - 20, h - 20, label, conf, color)
            detections.append((label, conf))
        else:
            # Klasifikasi setiap area daun yang terdeteksi
            for (x, y, bw, bh) in boxes:
                # Crop area daun
                crop = frame_rgb[y:y+bh, x:x+bw]
                if crop.size == 0:
                    continue
                
                # Klasifikasi
                label, conf, color, _ = classify_region(crop)
                
                # Gambar bounding box
                frame_rgb = draw_bounding_box(frame_rgb, x, y, bw, bh, label, conf, color)
                detections.append((label, conf))
    else:
        # Mode bounding box di seluruh gambar
        label, conf, color, _ = classify_region(frame_rgb)
        h, w = frame.shape[:2]
        # Beri padding 10px
        frame_rgb = draw_bounding_box(frame_rgb, 10, 10, w - 20, h - 20, label, conf, color)
        detections.append((label, conf))
    
    return frame_rgb, detections

# ==========================================
# MODE 1: UPLOAD GAMBAR
# ==========================================
if mode == "📷 Upload Gambar":
    st.header("📷 Klasifikasi Gambar Daun Mangrove")
    uploaded_file = st.file_uploader("Upload gambar daun mangrove", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with col1:
            st.subheader("Gambar Input")
            img_rgb_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb_input, channels="RGB", use_container_width=True)
        
        with col2:
            st.subheader("Hasil Klasifikasi")
            with st.spinner("Menganalisis..."):
                annotated_frame, detections = process_frame_with_bbox(img)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Tampilkan gambar dengan bounding box
            st.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Tampilkan detail deteksi
            st.markdown("### Hasil Deteksi:")
            for i, (label, conf) in enumerate(detections, 1):
                if label == UNKNOWN_LABEL:
                    st.error(f"Box {i}: ❌ {label} (Confidence: {conf:.2f}%)")
                else:
                    st.success(f"Box {i}: 🌿 {label} (Confidence: {conf:.2f}%)")
            
            # Probabilitas (untuk deteksi pertama atau whole image)
            if not use_leaf_detection or len(detections) > 0:
                with st.spinner("Menghitung probabilitas..."):
                    if use_leaf_detection and len(detections) > 0:
                        # Ambil box terbesar untuk probabilitas
                        boxes = detect_leaf_regions(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        if boxes:
                            largest_box = max(boxes, key=lambda b: b[2] * b[3])
                            x, y, bw, bh = largest_box
                            crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y:y+bh, x:x+bw]
                            _, _, _, results = classify_region(crop)
                        else:
                            _, _, _, results = classify_region(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        _, _, _, results = classify_region(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                st.markdown("### Probabilitas Semua Kelas:")
                mangrove_probs = {}
                unknown_prob = 0.0
                
                for cls_id, cls_name in model.names.items():
                    prob = results[0].probs.data[cls_id].item() * 100
                    if cls_name in VALID_MANGROVE_CLASSES:
                        mangrove_probs[cls_name] = prob
                    else:
                        unknown_prob += prob
                
                # Tentukan label tertinggi untuk highlight
                top_label = max(mangrove_probs, key=mangrove_probs.get) if mangrove_probs else UNKNOWN_LABEL
                if unknown_prob > max(mangrove_probs.values()) if mangrove_probs else True:
                    top_label = UNKNOWN_LABEL
                
                for cls_name, prob in mangrove_probs.items():
                    marker = " ⬅️ (Tertinggi)" if cls_name == top_label else ""
                    st.write(f"**{cls_name}**: {prob:.2f}%{marker}")
                    st.progress(prob / 100)
                
                marker = " ⬅️ (Tertinggi)" if top_label == UNKNOWN_LABEL else ""
                st.write(f"**{UNKNOWN_LABEL}**: {unknown_prob:.2f}%{marker}")
                st.progress(unknown_prob / 100)

# ==========================================
# MODE 2: UPLOAD VIDEO
# ==========================================
elif mode == "🎬 Upload Video":
    st.header("🎬 Klasifikasi Frame-by-Frame pada Video")
    uploaded_video = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        info_placeholder = st.empty()
        
        st.success("Video berhasil diupload. Memproses...")
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Proses frame dengan bounding box
            annotated_frame, detections = process_frame_with_bbox(frame)
            
            # Tampilkan frame
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # Update info
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Buat summary deteksi
            det_summary = ", ".join([f"{l} ({c:.0f}%)" for l, c in detections[:3]])
            if len(detections) > 3:
                det_summary += f", +{len(detections)-3} lainnya"
            
            info_placeholder.info(f"Frame: {frame_count} | FPS: {fps:.1f} | Deteksi: {det_summary}")
        
        cap.release()
        os.unlink(tfile.name)
        st.success("✅ Video selesai diproses!")

# ==========================================
# MODE 3: URL VIDEO
# ==========================================
elif mode == "🌐 URL Video":
    st.header("🌐 Klasifikasi Video dari URL")
    st.write("Masukkan URL video (mp4, streaming, atau CCTV). Contoh: https://example.com/video.mp4")
    
    video_url = st.text_input("URL Video:", placeholder="https://...")
    
    if st.button("Mulai Proses") and video_url:
        try:
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                st.error("❌ Gagal membuka video. Periksa URL atau format video.")
            else:
                stframe = st.empty()
                info_placeholder = st.empty()
                st.success("Video berhasil dimuat. Memproses...")
                
                frame_count = 0
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Proses frame dengan bounding box
                    annotated_frame, detections = process_frame_with_bbox(frame)
                    stframe.image(annotated_frame, channels="RGB", use_container_width=True)
                    
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    det_summary = ", ".join([f"{l} ({c:.0f}%)" for l, c in detections[:3]])
                    if len(detections) > 3:
                        det_summary += f", +{len(detections)-3} lainnya"
                    
                    info_placeholder.info(f"Frame: {frame_count} | FPS: {fps:.1f} | Deteksi: {det_summary}")
                
                cap.release()
                st.success("✅ Selesai!")
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# MODE 4: WEBCAM LIVE
# ==========================================
elif mode == "📹 Webcam Live":
    st.header("📹 Klasifikasi Real-time via Webcam")
    st.write("Aktifkan webcam untuk klasifikasi daun mangrove secara real-time.")
    st.info(f"💡 Threshold saat ini: **{confidence_threshold}%** | Auto-detect daun: **{'Aktif' if use_leaf_detection else 'Nonaktif'}**")
    
    run_camera = st.checkbox("Start / Stop Webcam")
    FRAME_WINDOW = st.image([])
    info_placeholder = st.empty()
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Gagal mengakses webcam. Pastikan kamera tidak digunakan aplikasi lain.")
        else:
            st.success("Webcam aktif! Arahkan kamera ke daun mangrove.")
            frame_count = 0
            start_time = time.time()
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal membaca frame dari webcam.")
                    break
                
                # Proses frame dengan bounding box
                annotated_frame, detections = process_frame_with_bbox(frame)
                FRAME_WINDOW.image(annotated_frame, channels="RGB", use_container_width=True)
                
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                det_summary = ", ".join([f"{l} ({c:.0f}%)" for l, c in detections[:3]])
                if len(detections) > 3:
                    det_summary += f", +{len(detections)-3} lainnya"
                
                info_placeholder.info(f"FPS: {fps:.1f} | Deteksi: {det_summary}")
            
            cap.release()
    else:
        st.write("Webcam dimatikan.")
