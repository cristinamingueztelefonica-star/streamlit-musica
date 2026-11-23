"""
Aplicació Streamlit: generació de música amb Facebook MusicGen Small.

Execució local:
    streamlit run app.py

L'aplicació permet generar un fragment musical a partir d'una descripció textual.
"""

import numpy as np
import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"


@st.cache_resource(show_spinner="Carregant el model MusicGen... Aquesta operació pot trigar uns minuts.")
def carregar_model():
    """
    Carrega el processador i el model una sola vegada.

    El decorador st.cache_resource conserva aquests objectes entre les
    execucions de l'aplicació per evitar descarregar el model repetidament.
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def generar_musica(descripcio, tokens, processor, model):
    """Genera un fragment musical a partir d'una descripció textual."""

    if not descripcio or not descripcio.strip():
        raise ValueError("Escriu una descripció de la música que vols generar.")

    # El processador transforma la descripció textual en tensors d'entrada.
    inputs = processor(
        text=[descripcio.strip()],
        padding=True,
        return_tensors="pt"
    )

    # Fem servir CUDA si està disponible; altrament, executem el model amb CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        # La generació és una operació d'inferència, no d'entrenament.
        with torch.inference_mode():
            audio = model.generate(
                **inputs,
                max_new_tokens=int(tokens)
            )

        # Convertim el resultat en un array NumPy perquè Streamlit el pugui reproduir.
        audio_array = audio[0, 0].detach().cpu().numpy()
        sample_rate = model.config.audio_encoder.sampling_rate

    finally:
        # Alliberem la GPU després de generar el fragment, quan sigui possible.
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sample_rate, audio_array


def main():
    """Construeix la interfície web de l'aplicació amb Streamlit."""

    st.set_page_config(
        page_title="Generador de música amb MusicGen",
        page_icon="🎵",
        layout="centered"
    )

    st.title("🎵 Generador de música amb MusicGen")
    st.write(
        "Descriu la música que vols escoltar i el model "
        "`facebook/musicgen-small` generarà un fragment d'àudio."
    )

    # Carreguem el processador i el model, reutilitzant la memòria cau.
    processor, model = carregar_model()

    # Camp de text on l'usuari descriu el tipus de música que vol generar.
    descripcio = st.text_area(
        "Descripció musical",
        value="melodic song with cello and flute",
        placeholder="Ex.: relaxing piano music with soft strings",
        height=110
    )

    # El nombre de tokens generats influeix en la durada del fragment
    # i en el temps i els recursos necessaris per generar-lo.
    tokens = st.slider(
        "Quantitat de tokens generats (afecta la durada i el cost)",
        min_value=64,
        max_value=512,
        value=256,
        step=64
    )

    # En prémer el botó, iniciem la generació de música.
    if st.button("Generar música", type="primary"):
        if not descripcio or not descripcio.strip():
            st.error("Escriu una descripció de la música que vols generar.")
        else:
            try:
                with st.spinner("Generant música... Aquest procés pot trigar una estona."):
                    sample_rate, audio_array = generar_musica(
                        descripcio,
                        tokens,
                        processor,
                        model
                    )

                st.success("La música s'ha generat correctament.")

                # Streamlit reprodueix l'àudio a partir de les dades i la freqüència
                # de mostreig retornades pel model.
                st.audio(audio_array, sample_rate=sample_rate)

            except Exception as error:
                st.error(f"No s'ha pogut generar la música: {error}")


if __name__ == "__main__":
    main()
