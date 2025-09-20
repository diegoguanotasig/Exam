import streamlit as st
from PIL import Image
import numpy as np
import easyocr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from openai import OpenAI
from huggingface_hub import login
import pandas as pd
import altair as alt

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;  /* color suave */
        background-image: url("https://st.depositphotos.com/2012693/1941/i/450/depositphotos_19416695-stock-photo-zen-path-of-stones-in.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    .stTextInput>div>input {
        background-color: rgba(255,255,255,0.8);
    }
    .stTextArea>div>textarea {
        background-color: rgba(255,255,255,0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)

#HF_TOKEN = "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"

@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(
        "AkDieg0/audit_distilbeto",
        use_auth_token=st.secrets["HF_TOKEN"]
        #use_auth_token=HF_TOKEN
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "AkDieg0/audit_distilbeto",
        use_auth_token=st.secrets["HF_TOKEN"]
        #use_auth_token=HF_TOKEN
    )
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
    st.image(img, caption="Imagen cargada", use_container_width=True)

    # OCR
    ocr_result = reader.readtext(np.array(img), detail=1)
    extracted_text = " ".join([r[1] for r in ocr_result])

    st.subheader("📄 Texto extraído")
    if extracted_text.strip() == "":
        st.warning("⚠️ No se detectó texto en la imagen. Verifica que la imagen contenga texto legible.")
    else:
        st.text_area("Texto detectado", extracted_text, height=150)
        if extracted_text.strip() != "":
            preds = classifier(extracted_text)
        st.subheader("📊 Clasificación")
    df_preds = pd.DataFrame(preds[0])

    # Mostramos tabla de probabilidades
    st.write("### Resultados detallados")
    st.dataframe(df_preds)

    # Gráfico de barras
    st.write("### Visualización de probabilidades")
    chart = (
        alt.Chart(df_preds)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Probabilidad"),
            y=alt.Y("label:N", sort="-x", title="Etiqueta"),
            color="label:N"
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Probabilidad específica de 'label_1' (recomendación)
    prob_recomend = df_preds.loc[df_preds["label"].str.lower() == "label_1", "score"].max()

    if pd.isna(prob_recomend):
        prob_recomend = 0.0

    st.write(f"🔎 Probabilidad de recomendación: **{prob_recomend:.2f}**")

    if prob_recomend >= threshold:
        st.success("✅ La imagen contiene una recomendación de auditoría.")   
       
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
    else:
        st.error("❌ No se detectó una recomendación de auditoría en la imagen")







