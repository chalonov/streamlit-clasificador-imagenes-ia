import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import cv2
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Imágenes IA",
    page_icon="🤖",
    layout="wide"
)

# Cache del modelo para evitar recargas
@st.cache_resource
def load_model():
    """Carga el modelo MobileNetV2 preentrenado"""
    model = MobileNetV2(weights='imagenet', include_top=True)
    return model

# Función para preprocesar imagen
def preprocess_image(img):
    """Preprocesa la imagen para el modelo"""
    # Redimensionar a 224x224 (tamaño requerido por MobileNetV2)
    img_resized = img.resize((224, 224))
    # Convertir a array numpy
    img_array = image.img_to_array(img_resized)
    # Expandir dimensiones para batch
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocesar según el modelo
    img_array = preprocess_input(img_array)
    return img_array

# Función para predecir
def predict_image(model, img_array):
    """Realiza la predicción sobre la imagen"""
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    return decoded_predictions

# Función para análisis de colores
def analyze_colors(img):
    """Analiza los colores dominantes en la imagen"""
    # Convertir PIL a numpy array
    img_array = np.array(img)
    
    # Redimensionar para acelerar el procesamiento
    img_small = cv2.resize(img_array, (100, 100))
    
    # Reshape para análisis de colores
    pixels = img_small.reshape(-1, 3)
    
    # Usar K-means para encontrar colores dominantes
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # Calcular porcentajes
    unique, counts = np.unique(labels, return_counts=True)
    percentages = (counts / len(labels)) * 100
    
    return colors, percentages

# Título principal
st.title("🤖 Clasificador de Imágenes con IA")
st.markdown("*Powered by MobileNetV2 - Modelo preentrenado en ImageNet*")
st.markdown("---")

# Sidebar para opciones
st.sidebar.title("⚙️ Opciones")
input_method = st.sidebar.selectbox(
    "Método de entrada:",
    ["Subir imagen", "Usar cámara", "URL de imagen", "Imágenes de ejemplo"]
)

# Cargar modelo
with st.spinner("Cargando modelo de IA..."):
    model = load_model()

st.sidebar.success("✅ Modelo cargado correctamente")

# Variable para almacenar la imagen
uploaded_image = None

# Diferentes métodos de entrada
if input_method == "Subir imagen":
    st.header("📁 Subir Imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen:",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)

elif input_method == "Usar cámara":
    st.header("📷 Capturar con Cámara")
    camera_image = st.camera_input("Toma una foto:")
    
    if camera_image is not None:
        uploaded_image = Image.open(camera_image)

elif input_method == "URL de imagen":
    st.header("🌐 Imagen desde URL")
    image_url = st.text_input(
        "Introduce la URL de la imagen:",
        placeholder="https://ejemplo.com/imagen.jpg"
    )
    
    if image_url:
        try:
            response = requests.get(image_url)
            uploaded_image = Image.open(BytesIO(response.content))
            st.success("✅ Imagen cargada desde URL")
        except Exception as e:
            st.error(f"❌ Error al cargar la imagen: {str(e)}")

elif input_method == "Imágenes de ejemplo":
    st.header("🖼️ Imágenes de Ejemplo")
    
    # URLs de imágenes de ejemplo
    example_images = {
        "Gato": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300",
        "Perro": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=300",
        "Coche": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=300",
        "Avión": "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=300",
        "Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=300"
    }
    
    selected_example = st.selectbox("Selecciona una imagen de ejemplo:", list(example_images.keys()))
    
    if st.button("🔄 Cargar imagen de ejemplo"):
        try:
            response = requests.get(example_images[selected_example])
            uploaded_image = Image.open(BytesIO(response.content))
            st.success(f"✅ Imagen de {selected_example} cargada")
        except Exception as e:
            st.error(f"❌ Error al cargar la imagen: {str(e)}")

# Procesar imagen si está disponible
if uploaded_image is not None:
    # Mostrar imagen
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Imagen Original")
        st.image(uploaded_image, use_column_width=True)
        
        # Información de la imagen
        st.info(f"""
        **Información de la imagen:**
        - Dimensiones: {uploaded_image.size}
        - Modo: {uploaded_image.mode}
        - Formato: {getattr(uploaded_image, 'format', 'Desconocido')}
        """)
    
    with col2:
        st.subheader("🤖 Análisis con IA")
        
        if st.button("🔍 Clasificar Imagen", type="primary"):
            with st.spinner("Analizando imagen..."):
                # Preprocesar imagen
                processed_img = preprocess_image(uploaded_image)
                
                # Realizar predicción
                predictions = predict_image(model, processed_img)
                
                # Mostrar resultados
                st.success("✅ Análisis completado")
                
                # Crear DataFrame con resultados
                results_data = []
                for i, (imagenet_id, label, confidence) in enumerate(predictions):
                    results_data.append({
                        'Posición': i + 1,
                        'Clase': label.replace('_', ' ').title(),
                        'Confianza': f"{confidence * 100:.2f}%",
                        'Confianza_num': confidence * 100
                    })
                
                df_results = pd.DataFrame(results_data)
                
                # Mostrar tabla de resultados
                st.subheader("📊 Resultados de Clasificación")
                st.dataframe(df_results[['Posición', 'Clase', 'Confianza']], use_container_width=True)
                
                # Gráfico de barras con confianzas
                fig_bar = px.bar(
                    df_results,
                    x='Confianza_num',
                    y='Clase',
                    orientation='h',
                    title="Niveles de Confianza por Clase",
                    labels={'Confianza_num': 'Confianza (%)', 'Clase': 'Clase Predicha'},
                    color='Confianza_num',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Mostrar la predicción principal
                best_prediction = predictions[0]
                st.markdown(f"""
                ### 🎯 Predicción Principal
                **{best_prediction[1].replace('_', ' ').title()}**  
                *Confianza: {best_prediction[2] * 100:.2f}%*
                """)
    
    # Análisis adicional de colores
    st.markdown("---")
    st.subheader("🎨 Análisis de Colores Dominantes")
    
    if st.button("🔍 Analizar Colores"):
        with st.spinner("Analizando colores..."):
            try:
                colors, percentages = analyze_colors(uploaded_image)
                
                # Crear visualización de colores
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Mostrar paleta de colores
                    st.write("**Colores Dominantes:**")
                    for i, (color, percentage) in enumerate(zip(colors, percentages)):
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <div style="width: 30px; height: 30px; background-color: rgb({color[0]}, {color[1]}, {color[2]}); 
                                           border: 1px solid #ccc; margin-right: 10px;"></div>
                                <span>RGB({color[0]}, {color[1]}, {color[2]}) - {percentage:.1f}%</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                with col2:
                    # Gráfico de pie con colores
                    color_df = pd.DataFrame({
                        'Color': [f'Color {i+1}' for i in range(len(colors))],
                        'Porcentaje': percentages,
                        'RGB': [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colors]
                    })
                    
                    fig_pie = px.pie(
                        color_df,
                        values='Porcentaje',
                        names='Color',
                        title="Distribución de Colores",
                        color_discrete_sequence=[rgb for rgb in color_df['RGB']]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error en análisis de colores: {str(e)}")

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Información del Modelo")
st.sidebar.info("""
**MobileNetV2**
- Modelo preentrenado en ImageNet
- 1000 clases diferentes
- Optimizado para dispositivos móviles
- Precisión: ~71% Top-1 en ImageNet
""")

st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Usa imágenes claras y bien iluminadas
- El modelo funciona mejor con objetos centrados
- Prueba diferentes ángulos para mejor precisión
- Algunas clases pueden tener nombres técnicos
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🤖 <strong>Clasificador de Imágenes con IA</strong> | 
        Modelo: MobileNetV2 | Dataset: ImageNet</p>
        <p><em>Esta aplicación demuestra capacidades básicas de visión computacional</em></p>
    </div>
    """,
    unsafe_allow_html=True
)