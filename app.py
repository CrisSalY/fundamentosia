import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import pandas as pd 
from openai import OpenAI 
from collections import Counter

api_key = st.secrets["API_KEY"]

#config  
page_bg_color = "#f0f8ff"   
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {page_bg_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
) 
st.set_page_config(layout="wide")
#cargar el modelo
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error(f"No se encontró el modelo en {model_path}")
else:
    model = YOLO(model_path) 
st.title("Detección de Enfermedades en rosas")
st.subheader("Métricas del modelo YOLO personalizado")
#cargar agente IA
os.environ['OPENAI_API_KEY'] = api_key
#funcion para responder preguntas del modo adecuado
def preguntar(question):
    sys_cont="""
    You are a professional consultant specializing in flower cultivation in Ecuador, particularly roses.
    Your purpose is to help mitigate rose diseases that may affect large-scale or home production.
    You specialize in combating powdery mildew, downy mildew, and black spots in industrial or home cultivation.
    You will be asked about how to combat powdery mildew, downy mildew, and black spots, and you should assume this is
    on a large scale unless otherwise specified. You must provide clear information aimed at people knowledgeable about flowers. 
    Your answers should be concise and to the point to mitigate the disease mentioned.
    You only answer questions and give care advice about plants and flowers; any other questions should not be answered but 
    rather redirected to the topic of rose diseases.
    When mentioning products such as flower chemicals, you must include the product name, market availability, price, risks, etc.
    """ 
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":sys_cont },
            {"role": "user", "content":'tengo esta planta con la enfermedad: moho oidico; en un cultivo a gran escala. Que debo hacer para mitigar esta enfermedad?' },
            {"role": "assistant", "content":'Hola, el moho oidico afecta a la rosa de esta manera, se combate de esta forma, se usan estos productos, se aplican de esta forma, cuestan este precio y aqui hay algunas recomendaciones para cuidar tus rosas.'},
            {"role": "user", "content": question}
        ]
    ) 
    return response.choices[0].message.content
#header: metricas +imagenes
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])  
metrics_path = "results.csv"
best_metrics = None 
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path) 
    best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()] 
    metrics_table = pd.DataFrame({
        "Métrica": [ "mAP50",  "Precisión", "Recall", "Box Loss","mAP50-95",],
        "Valor": [ 
            round(best_row['metrics/mAP50(B)'], 4),
            round(best_row['metrics/precision(B)'], 4),
            round(best_row['metrics/recall(B)'], 4),
            round(best_row['train/box_loss'], 4),
            round(best_row['metrics/mAP50-95(B)'], 4)
        ]
    })  
    with col1:   
        st.markdown(
            metrics_table.to_html(index=False, justify='center'),
            unsafe_allow_html=True )
else:
    st.info("No se encontraron métricas detalladas.")
img1 = Image.open("imagen1.png")
img2 = Image.open("imagen2.jpeg")
img3 = Image.open("imagen3.png")
with col2:
    st.image(img1, caption="Moho Oídio",width=230)
with col3:
    st.image(img2, caption="Moho Mildiu",width=218)
with col4:
    st.image(img3, caption="Manchas negras",width=230)
#foto
st.subheader("Detecta enfermedades en una foto")
col1_1, col2_1= st.columns([1, 1])  
with col1_1:
    uploaded_image = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagen subida", width=250 )        
with col2_1:
    col21, col22,  = st.columns([1, 1])  
    detected_classes=[]
    with col21:
        # Barra deslizante para la confianza
        if uploaded_image is not None: 
            conf_threshold = st.slider(
            "Nivel de confianza",  # Etiqueta
            min_value=0.0,         # Valor mínimo
            max_value=1.0,         # Valor máximo
            value=0.5,             # Valor inicial
            step=0.01,              # Incremento
            width=350
            )
            results = model.predict(image, imgsz=640, conf=conf_threshold)
            for box in results[0].boxes:
                        cls_id = int(box.cls)
                        detected_classes.append(results[0].names[cls_id])
            annotated_img = results[0].plot()  
            st.image(annotated_img, caption="Detecciones",width=350 )  
            annotated_img_pil = Image.fromarray(annotated_img)
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            annotated_img_pil.save(temp_img.name)
    with col22:
        if uploaded_image is not None:  
            st.download_button("Descargar imagen procesada", data=open(temp_img.name, "rb"), file_name="imagen_detectada.jpg")
            st.write("Enfermedades detectadas en la foto:")
            if detected_classes:
                counts = Counter(detected_classes)
                for cls, count in counts.items():
                    st.write(f"- {cls}: {count} veces")
            else:
                st.write("No se detectaron enfermedades en las rosas.")
            question = 'tengo esta planta con las enfermedades:'+', '.join(list(set(detected_classes)))+'; en un cultivo a gran escala. Que debo hacer para mitigar esta enfermedad?' 
            st.write(preguntar(question))
        
#video
st.subheader("Detecta enfermedades en un video")
col1, col2= st.columns([1, 1])  
with col1:
    uploaded_video = st.file_uploader("Selecciona un video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None: 
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        st.video(temp_video.name,width=350,muted=True)
with col2:
    if uploaded_video is not None:          
        conf_threshold = st.slider(
            "Nivel de confianza",  # Etiqueta
            min_value=0.0,         # Valor mínimo
            max_value=1.0,         # Valor máximo
            value=0.5,             # Valor inicial
            step=0.01,              # Incremento
            width=350,
            key='video'
        )  
        if st.button("Procesar video"):     
            st.write(f"Procesando video con confianza: {conf_threshold}")
            cap = cv2.VideoCapture(temp_video.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            max_frames = fps * 3  
            frame_count = 0   
            progress = int((frame_count / max_frames) * 100)
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            ret, frame = cap.read()
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            progress_bar = st.progress(0)
            status_text = st.empty() 
            detected_classes = []
            while cap.isOpened() and frame_count < max_frames:                
                ret, frame = cap.read()
                if not ret:
                    break       
                results = model.predict(frame, imgsz=640, conf=conf_threshold)
                annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)   
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    detected_classes.append(results[0].names[cls_id])
                out.write(annotated_frame) 
                frame_count += 1
                progress = int((frame_count / max_frames) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Procesando frame {frame_count}/{max_frames} ({progress}%)")
            cap.release()
            out.release() 
            st.success("Detección completada.")             
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Descargar video procesado",
                    data=file,
                    file_name="video_detectado.mp4",
                    mime="video/mp4") 
            st.write("Enfermedades detectadas en el video:")
            if detected_classes:
                counts = Counter(detected_classes)
                for cls, count in counts.items():
                    st.write(f"- {cls}: {count} veces")
            else:
                st.write("No se detectaron enfermedades en las rosas.")
            question = 'tengo esta planta con las enfermedades:'+', '.join(list(set(detected_classes)))+'; en un cultivo a gran escala. Que debo hacer para mitigar esta enfermedad?' 
            st.write(preguntar(question))

