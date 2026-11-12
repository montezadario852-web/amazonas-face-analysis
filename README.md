
# AMAZONAS FACE ANALYSIS

Prototipo académico de análisis geométrico facial con Python, Streamlit y MediaPipe.

## Ejecutar en Windows

1. Instala Python 3.11 (64 bits).
2. Descomprime esta carpeta.
3. Abre PowerShell dentro de la carpeta del proyecto.
4. Ejecuta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. Se abrirá la aplicación en el navegador. Si no se abre automáticamente, visita:
   http://localhost:8501

La primera ejecución necesita conexión a internet para descargar el modelo oficial de MediaPipe.

## Publicar y obtener un enlace web

Para tener un enlace público:
1. Crea una cuenta en GitHub.
2. Crea un repositorio nuevo y sube `app.py` y `requirements.txt`.
3. Entra a https://share.streamlit.io/
4. Conecta GitHub, elige el repositorio y selecciona `app.py` como archivo principal.
5. Pulsa Deploy. Streamlit Community Cloud generará la URL pública de tu aplicación.

## Alcance y uso responsable

Esta herramienta calcula medidas geométricas de imágenes faciales y permite explorar datos de una muestra voluntaria. No determina si una persona es amazónica, indígena, ni su origen étnico o geográfico. La procedencia de la muestra debe registrarse mediante autodeclaración voluntaria y consentimiento informado. Evita guardar fotografías o datos identificables sin autorización expresa.

## Variables geométricas incluidas

- Relación ancho facial / alto facial
- Distancia entre ojos / ancho facial
- Ancho de nariz / ancho facial
- Ancho de boca / ancho facial
- Ancho de mandíbula / ancho facial
- Longitud nasal / alto facial

Las medidas son exploratorias y dependen de pose, iluminación, expresión y calidad de imagen.
