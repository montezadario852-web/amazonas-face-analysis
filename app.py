

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import urllib.request
from pathlib import Path

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================

st.set_page_config(
    page_title="Amazonas Face Analysis",
    page_icon="📐",
    layout="wide"
)

MODEL_PATH = Path("face_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

# ==========================================
# CARGA DEL MODELO FACIAL
# ==========================================

@st.cache_resource
def load_landmarker():

    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


# ==========================================
# LECTURA DE IMÁGENES
# ==========================================

def read_image(uploaded_file):

    data = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================
# CÁLCULO DE DISTANCIAS
# ==========================================

def distance(a, b):

    return float(
        np.linalg.norm(
            np.array(a[:2]) - np.array(b[:2])
        )
    )


# ==========================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================

def extract_features(landmarks):

    points = [
        (lm.x, lm.y, lm.z)
        for lm in landmarks
    ]

    face_width = distance(
        points[234], points[454]
    )

    face_height = distance(
        points[10], points[152]
    )

    eye_distance = distance(
        points[33], points[263]
    )

    nose_width = distance(
        points[129], points[358]
    )

    mouth_width = distance(
        points[61], points[291]
    )

    nose_length = distance(
        points[1], points[2]
    )

    jaw_width = distance(
        points[172], points[397]
    )

    return {
        "Ancho facial / alto facial":
            face_width / max(face_height, 1e-8),

        "Distancia entre ojos / ancho facial":
            eye_distance / max(face_width, 1e-8),

        "Ancho de nariz / ancho facial":
            nose_width / max(face_width, 1e-8),

        "Ancho de boca / ancho facial":
            mouth_width / max(face_width, 1e-8),

        "Ancho de mandíbula / ancho facial":
            jaw_width / max(face_width, 1e-8),

        "Longitud nasal / alto facial":
            nose_length / max(face_height, 1e-8)
    }


# ==========================================
# DIBUJAR PUNTOS FACIALES
# ==========================================

def draw_landmarks(image, landmarks):

    height, width = image.shape[:2]

    output = image.copy()

    for lm in landmarks:

        x = int(lm.x * width)
        y = int(lm.y * height)

        if 0 <= x < width and 0 <= y < height:

            cv2.circle(
                output,
                (x, y),
                1,
                (40, 220, 100),
                -1
            )

    return output


# ==========================================
# ANÁLISIS FACIAL
# ==========================================

def analyze(image):

    rgb = np.ascontiguousarray(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    landmarker = load_landmarker()

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:

        return None, None

    landmarks = result.face_landmarks[0]

    features = extract_features(
        landmarks
    )

    annotated = draw_landmarks(
        image,
        landmarks
    )

    return features, annotated


# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

st.title("📐 AMAZONAS FACE ANALYSIS")

st.subheader(
    "Análisis y comparación geométrica de rostros"
)

st.markdown(
    """
    Sistema experimental desarrollado en Python
    para estudiar características geométricas faciales
    mediante visión artificial.
    """
)

st.warning(
    """
    IMPORTANTE: Este sistema analiza proporciones
    geométricas de imágenes. No determina el origen
    étnico ni la procedencia geográfica de una persona.
    """
)

# ==========================================
# PANEL LATERAL
# ==========================================

with st.sidebar:

    st.header("⚙️ Panel de análisis")

    st.write("Versión: Prototipo académico 1.1")

    st.write("Motor: MediaPipe Face Landmarker")

    st.write("Método: Proporciones geométricas")

    st.divider()

    st.info(
        "Las fotografías se procesan durante la sesión. "
        "No se guardan automáticamente."
    )


# ==========================================
# PESTAÑAS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📷 Analizar rostro",
    "📊 Comparar muestra",
    "ℹ️ Metodología"
])


# ==========================================
# PESTAÑA 1: ANÁLISIS FACIAL
# ==========================================

with tab1:

    st.header("Análisis de características faciales")

    uploaded = st.file_uploader(
        "Selecciona una fotografía JPG, JPEG o PNG",
        type=["jpg", "jpeg", "png"],
        key="face_upload"
    )

    if uploaded is not None:

        image = read_image(uploaded)

        if image is None:

            st.error(
                "No se pudo leer la fotografía."
            )

        else:

            try:

                with st.spinner(
                    "Analizando características geométricas..."
                ):

                    features, annotated = analyze(image)

                if features is None:

                    st.error(
                        "No se detectó ningún rostro. "
                        "Utiliza una fotografía frontal "
                        "con buena iluminación."
                    )

                else:

                    st.success(
                        "Rostro detectado correctamente."
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.image(
                            image,
                            caption="Fotografía original",
                            use_container_width=True
                        )

                    with col2:

                        st.image(
                            annotated,
                            caption="Puntos faciales detectados",
                            use_container_width=True
                        )

                    st.divider()

                    st.subheader(
                        "📏 Mediciones geométricas"
                    )

                    df = pd.DataFrame({
                        "Característica": list(features.keys()),
                        "Valor normalizado": list(features.values())
                    })

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader(
                        "📊 Representación gráfica"
                    )

                    chart_df = df.set_index(
                        "Característica"
                    )

                    st.bar_chart(chart_df)

                    csv = df.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="⬇️ Descargar resultados CSV",
                        data=csv,
                        file_name="mediciones_faciales.csv",
                        mime="text/csv"
                    )

            except Exception as error:

                st.error(
                    f"Error durante el análisis: {error}"
                )

                st.info(
                    "Comprueba las dependencias y la conexión "
                    "necesaria para descargar el modelo facial."
                )


# ==========================================
# PESTAÑA 2: COMPARACIÓN DE MUESTRAS
# ==========================================

with tab2:

    st.header(
        "Comparación estadística de una muestra"
    )

    st.write(
        """
        Carga un archivo CSV que contenga mediciones
        geométricas de imágenes de referencia obtenidas
        con consentimiento de los voluntarios.
        """
    )

    sample = st.file_uploader(
        "Seleccionar archivo CSV",
        type=["csv"],
        key="sample_upload"
    )

    if sample is not None:

        try:

            data = pd.read_csv(sample)

            st.subheader(
                "Datos de la muestra"
            )

            st.dataframe(
                data.head(20),
                use_container_width=True
            )

            numeric = data.select_dtypes(
                include=np.number
            )

            if numeric.empty:

                st.warning(
                    "No se encontraron columnas numéricas."
                )

            else:

                st.subheader(
                    "Resumen estadístico"
                )

                st.dataframe(
                    numeric.describe().T,
                    use_container_width=True
                )

                st.subheader(
                    "Promedios de las características"
                )

                st.bar_chart(
                    numeric.mean()
                )

        except Exception as error:

            st.error(
                f"No se pudo leer el CSV: {error}"
            )

    else:

        st.info(
            "Todavía no has cargado una muestra de referencia."
        )


# ==========================================
# PESTAÑA 3: METODOLOGÍA
# ==========================================

with tab3:

    st.header(
        "Fundamento matemático y metodológico"
    )

    st.markdown(
        """
        ### 1. Detección facial

        Se utiliza MediaPipe Face Landmarker para
        localizar puntos de referencia del rostro.

        ### 2. Extracción de características

        Se identifican puntos faciales para calcular
        distancias entre regiones anatómicas.

        ### 3. Modelo matemático

        La distancia euclidiana entre dos puntos
        se calcula mediante:

        \\[
        d(A,B)=\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
        \\]

        ### 4. Normalización

        Las distancias se dividen entre dimensiones
        faciales para obtener proporciones geométricas.

        ### 5. Comparación estadística

        Las mediciones de una muestra voluntaria
        pueden analizarse mediante medias, desviaciones
        estándar y distribuciones estadísticas.

        ### 6. Limitaciones

        Los resultados pueden variar según la pose,
        iluminación, expresión facial y calidad de imagen.

        Este prototipo no identifica personas ni demuestra
        origen étnico o procedencia geográfica.
        """
    )

    st.success(
        "Proyecto académico: Amazonas Face Analysis."
    )

    st.caption(
        "Desarrollado con Python, OpenCV, NumPy, "
        "Pandas y MediaPipe."
    )


# ==========================================
# PIE DE PÁGINA
# ==========================================

st.divider()

st.caption(
    "AMAZONAS FACE ANALYSIS | Prototipo académico"
)

