import streamlit as st
from PIL import Image
import numpy as np
import easyocr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from openai import OpenAI
from huggingface_hub import login

@st.cache_resource
def load_classifier():
    login(token=st.secrets["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained("AkDieg0/audit_distilbeto")
    model = AutoModelForSequenceClassification.from_pretrained("AkDieg0/audit_distilbeto")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['es'])

classifier = load_classifier()
reader = load_ocr()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🔍 Identificador de Recomendaciones de Auditoría")

uploaded = st.file_uploader("Sube una imagen", type=['png','jpg','jpeg'])
threshold = st.slider("Umbral probabilidad", 0.0, 1.0, 0.6)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagen cargada", use_column_width=True)

    # OCR
    ocr_result = reader.readtext(np.array(img), detail=1)
    extracted_text = " ".join([r[1] for r in ocr_result])

    st.subheader("📄 Texto extraído")
    st.text_area("Texto detectado", extracted_text, height=150)

    if extracted_text.strip() != "":
        preds = classifier(extracted_text[:512])
        prob_recomend = 0.0
        for item in preds[0]:
            if item['label'].lower() in ["label_1"]:  # label_1 = recomendación
                prob_recomend = item['score']

        st.subheader("📊 Clasificación")
        st.write(f"Probabilidad de recomendación: **{prob_recomend:.2f}**")

        if prob_recomend >= threshold:
            if st.button("Generar actividades con OpenAI"):
               
                prompt = f"""
                Eres un asistente experto en auditoría.
                Recomendación: "{extracted_text}"

                Genera una lista de actividades concretas, con:
                - Responsable
                - Pasos principales
                - Tiempo estimado
                - Indicadores de cumplimiento
                - Riesgos y controles
                """

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"Eres un experto en auditoría."},
                        {"role":"user","content":prompt}
                    ],
                    max_tokens=600
                )
                suggestion = resp.choices[0].message.content
                st.subheader("✅ Actividades sugeridas")

                st.markdown(suggestion)

