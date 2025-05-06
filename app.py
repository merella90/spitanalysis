import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

st.set_page_config(page_title="Analisi Canali Agoda e WebBeds", layout="wide")
st.title("📊 Analisi Performance Ca' di Dio & Opportunità Agoda / WebBeds")

# Caricamento file dati
uploaded_file = st.file_uploader("Carica il file Excel con i dati attuali vs SPIT", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Parsing colonne attese
    col_date = st.selectbox("Seleziona la colonna Data", df.columns)
    col_occ_2025 = st.selectbox("Seleziona la colonna Occupazione 2025 (%)", df.columns)
    col_occ_2024 = st.selectbox("Seleziona la colonna Occupazione 2024 (SPIT) (%)", df.columns)
    col_adr_2025 = st.selectbox("Seleziona la colonna ADR 2025", df.columns)
    col_adr_2024 = st.selectbox("Seleziona la colonna ADR 2024 (SPIT)", df.columns)

    df['Data'] = pd.to_datetime(df[col_date])
    df['Mese'] = df['Data'].dt.strftime('%B')
    df['Giorno'] = df['Data'].dt.day_name()

    # Calcolo medie mensili e settimanali
    monthly = df.groupby('Mese').agg({
        col_occ_2025: 'mean',
        col_occ_2024: 'mean',
        col_adr_2025: 'mean',
        col_adr_2024: 'mean'
    }).round(1).reset_index()

    weekday = df.groupby('Giorno').agg({
        col_occ_2025: 'mean',
        col_occ_2024: 'mean'
    }).round(1).reset_index()

    weekday['Delta Occ. (%)'] = (weekday[col_occ_2025] - weekday[col_occ_2024]).round(1)

    st.subheader("📅 Performance mensile: 2025 vs SPIT 2024")
    st.dataframe(monthly)

    st.subheader("📆 Performance per giorno della settimana")
    st.dataframe(weekday.sort_values(by='Delta Occ. (%)'))

    # Conclusione automatica
    st.subheader("💡 Conclusione automatica")
    mean_occ_2025 = df[col_occ_2025].mean()
    mean_occ_2024 = df[col_occ_2024].mean()
    mean_adr_2025 = df[col_adr_2025].mean()
    mean_adr_2024 = df[col_adr_2024].mean()

    delta_occ = mean_occ_2025 - mean_occ_2024
    delta_adr = mean_adr_2025 - mean_adr_2024

    if delta_occ < -10:
        occ_text = "📉 L'occupazione è in forte calo rispetto al 2024 ({} punti % in meno).".format(round(abs(delta_occ), 1))
    else:
        occ_text = "📊 L'occupazione è relativamente stabile rispetto al 2024."

    if delta_adr < 0:
        adr_text = "💸 L'ADR medio è sceso di circa €{:.2f} rispetto al 2024.".format(abs(delta_adr))
    else:
        adr_text = "💰 L'ADR medio è aumentato di circa €{:.2f} rispetto al 2024.".format(delta_adr)

    st.markdown(f"**{occ_text}**\n\n**{adr_text}**")

    st.markdown("""
    ### 🔍 Raccomandazione:
    - Attivare **Agoda** per migliorare visibilità nei mercati asiatici e mobile.
    - Attivare **WebBeds** per accedere a tour operator e agenzie B2B in Europa.
    - Monitorare ADR e occupazione per evitare cannibalizzazione.
    - Mantenere il posizionamento luxury con disponibilità e tariffe coerenti.
    """)

else:
    st.info("Carica un file per iniziare l'analisi.")
