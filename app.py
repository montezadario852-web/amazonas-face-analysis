
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Amazonas Face Analysis", page_icon="📐", layout="wide")

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

@st.cache_resource
def load_landmarker():
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return FaceLandmarker.create_from_options(options)

def read_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def distance(a, b):
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))

def extract_features(landmarks):
    # Índices de referencia de MediaPipe Face Mesh; mediciones normalizadas
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
    face_width = distance(pts[234], pts[454])
    face_height = distance(pts[10], pts[152])
    eye_distance = distance(pts[33], pts[263])
    nose_width = distance(pts[129], pts[358])
    mouth_width = distance(pts[61], pts[291])
    nose_length = distance(pts[1], pts[2])
    jaw_width = distance(pts[172], pts[397])
    denom = max(face_width, 1e-8)
    return {
        "Ancho facial / alto facial": face_width / max(face_height, 1e-8),
        "Distancia entre ojos / ancho facial": eye_distance / denom,
        "Ancho de nariz / ancho facial": nose_width / denom,
        "Ancho de boca / ancho facial": mouth_width / denom,
        "Ancho de mandíbula / ancho facial": jaw_width / denom,
        "Longitud nasal / alto facial": nose_length / max(face_height, 1e-8)
    }

def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    out = image.copy()
    for lm in landmarks:
        x, y = int(lm.x*w), int(lm.y*h)
        cv2.circle(out, (x, y), 1, (40, 220, 100), -1)
    return out

def analyze(image):
    rgb = np.ascontiguousarray(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = load_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None, None
    landmarks = result.face_landmarks[0]
    return extract_features(landmarks), draw_landmarks(image, landmarks)

st.title("AMAZONAS FACE ANALYSIS")
st.caption("Sistema experimental para el análisis y comparación geométrica de rostros")
st.warning(
    "Este sistema mide características geométricas de imágenes. No determina ni demuestra "
    "origen étnico o procedencia geográfica. La pertenencia a una muestra de investigación "
    "debe registrarse mediante información voluntaria y consentimiento."
)

with st.sidebar:
    st.header("Panel de análisis")
    st.write("**Versión:** Prototipo académico 1.0")
    st.write("**Motor:** MediaPipe Face Landmarker")
    st.write("**Método:** proporciones y distancias normalizadas")
    st.divider()
    st.write("Las fotografías se procesan durante la sesión. No se guardan automáticamente.")

tab1, tab2, tab3 = st.tabs(["📷 Analizar rostro", "📊 Comparar muestra", "ℹ️ Metodología"])

with tab1:
    st.subheader("Captura o carga de fotografía")
    source = st.radio("Selecciona cómo ingresar la imagen", ["Tomar fotografía con cámara", "Subir fotografía"], horizontal=True)
    uploaded = None
    if source == "Tomar fotografía con cámara":
        uploaded = st.camera_input("Toma una fotografía frontal")
    else:
        uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = read_image(uploaded)
        if image is None:
            st.error("No se pudo leer la imagen. Prueba con JPG o PNG.")
        else:
            with st.spinner("Detectando puntos faciales y calculando medidas..."):
                try:
                    features, annotated = analyze(image)
                    if features is None:
                        st.error("No se detectó un rostro. Prueba con una fotografía frontal, bien iluminada y sin obstrucciones.")
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(image, caption="Fotografía ingresada", use_container_width=True)
                        with c2:
                            st.image(annotated, caption="Puntos faciales detectados", use_container_width=True)
                        st.success("Rostro detectado. Se calcularon las características geométricas.")
                        df = pd.DataFrame({"Característica": list(features.keys()), "Valor normalizado": list(features.values())})
                        st.subheader("Mediciones geométricas")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.bar_chart(df.set_index("Característica"))
                        st.download_button("Descargar mediciones CSV", df.to_csv(index=False).encode("utf-8"), "mediciones_faciales.csv", "text/csv")
                except Exception as e:
                    st.error(f"No se pudo completar el análisis: {e}")
                    st.info("Comprueba tu conexión a internet la primera vez para descargar el modelo facial.")

with tab2:
    st.subheader("Comparación geométrica de una muestra")
    st.write(
        "Carga un CSV con mediciones geométricas previamente obtenidas de voluntarios que "
        "hayan consentido participar. Cada fila debe representar una imagen codificada y "
        "contener las mismas variables que aparecen en la pestaña de análisis."
    )
    st.caption("No incluyas nombres, documentos, datos sensibles ni información que permita identificar a los voluntarios.")
    sample = st.file_uploader("Cargar CSV de muestra (opcional)", type=["csv"], key="sample")
    if sample:
        try:
            data = pd.read_csv(sample)
            st.dataframe(data.head(20), use_container_width=True)
            numeric = data.select_dtypes(include=np.number)
            if numeric.empty:
                st.warning("El archivo no contiene columnas numéricas para comparar.")
            else:
                st.write("Resumen estadístico descriptivo de la muestra")
                st.dataframe(numeric.describe().T, use_container_width=True)
                st.bar_chart(numeric.mean())
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
    else:
        st.info("Aún no hay una muestra cargada. Primero reúne y procesa imágenes de referencia con consentimiento.")

with tab3:
    st.subheader("Fundamento del procedimiento")
    st.markdown("""
    1. **Detección:** MediaPipe Face Landmarker localiza puntos de referencia faciales.
    2. **Extracción:** se calculan distancias euclidianas entre puntos seleccionados.
    3. **Normalización:** las distancias se dividen entre dimensiones faciales para reducir el efecto del tamaño de la imagen.
    4. **Descripción estadística:** una muestra voluntaria puede resumirse mediante medias, desviaciones estándar y distribuciones.
    5. **Comparación:** las mediciones permiten estudiar similitudes geométricas dentro de la muestra, sin inferir identidad, etnia ni procedencia.

    **Limitaciones:** los resultados varían con la pose, iluminación, expresión, cámara y calidad de imagen. Este prototipo no es un instrumento de diagnóstico ni de identificación.
    """)
    st.write("Para una investigación rigurosa, documenta el consentimiento, los criterios de inclusión, la calidad de las imágenes y el protocolo de captura.")
