import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

st.set_page_config(page_title="Analisi CDD", layout="wide")
st.title("📊 Analisi Performance CDD CY VS SPIT")

# Caricamento file dati
uploaded_file = st.file_uploader("Carica il file Excel con i dati attuali vs SPIT", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Parsing colonne attese
    col_date = st.selectbox("Seleziona la colonna Data", df.columns)
    col_occ_2025 = st.selectbox("Seleziona la colonna Occupazione 2025 (%)", df.columns)
    col_occ_diff = st.selectbox("Seleziona la colonna Differenza Occupazione vs SPIT", df.columns)
    col_adr_2025 = st.selectbox("Seleziona la colonna ADR 2025", df.columns)
    col_adr_diff = st.selectbox("Seleziona la colonna Differenza ADR vs SPIT", df.columns)

    df['Data'] = pd.to_datetime(df[col_date], errors='coerce')
    df = df.dropna(subset=['Data'])
    df['Mese'] = df['Data'].dt.strftime('%B')
    df['Mese'] = pd.Categorical(df['Mese'], categories=["May", "June", "July", "August", "September", "October"], ordered=True)
    df['Giorno'] = df['Data'].dt.day_name()
    df['Giorno'] = pd.Categorical(df['Giorno'], categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)

    # Conversione delle percentuali e calcolo Occupazione 2024 (SPIT)
    df[col_occ_2025] = df[col_occ_2025].astype(str).str.replace('%', '').str.replace(',', '.').astype(float) * 100
    df[col_occ_diff] = df[col_occ_diff].astype(float) * 100
    df['Occupazione 2024 (SPIT)'] = df[col_occ_2025] - df[col_occ_diff]

    # Conversione ADR e calcolo ADR SPIT
    df[col_adr_2025] = df[col_adr_2025].astype(str).str.replace(',', '.').astype(float)
    df[col_adr_diff] = df[col_adr_diff].astype(str).str.replace(',', '.').astype(float)
    df['ADR 2024 (SPIT)'] = df[col_adr_2025] - df[col_adr_diff]

    # Calcolo medie mensili e settimanali
    monthly = df.groupby('Mese').agg({
        col_occ_2025: 'mean',
        'Occupazione 2024 (SPIT)': 'mean',
        col_adr_2025: 'mean',
        'ADR 2024 (SPIT)': 'mean'
    }).reset_index().sort_values(by='Mese')

    weekday = df.groupby('Giorno').agg({
        col_occ_2025: 'mean',
        'Occupazione 2024 (SPIT)': 'mean'
    }).reset_index().sort_values(by='Giorno')

    weekday['Delta Occ. (%)'] = (weekday[col_occ_2025] - weekday['Occupazione 2024 (SPIT)'])

    # Formattazione percentuali per la visualizzazione
    monthly[col_occ_2025] = monthly[col_occ_2025].apply(lambda x: f"{x:.1f}%")
    monthly['Occupazione 2024 (SPIT)'] = monthly['Occupazione 2024 (SPIT)'].apply(lambda x: f"{x:.1f}%")

    weekday[col_occ_2025] = weekday[col_occ_2025].apply(lambda x: f"{x:.1f}%")
    weekday['Occupazione 2024 (SPIT)'] = weekday['Occupazione 2024 (SPIT)'].apply(lambda x: f"{x:.1f}%")
    weekday['Delta Occ. (%)'] = weekday['Delta Occ. (%)'].apply(lambda x: f"{x:.1f}%")

    st.subheader("📅 Performance mensile: 2025 vs SPIT 2024")
    st.dataframe(monthly)

    st.subheader("📆 Performance per giorno della settimana")
    st.dataframe(weekday)

    # Conclusione automatica
    mean_occ_2025 = df[col_occ_2025].mean()
    mean_occ_2024 = df['Occupazione 2024 (SPIT)'].mean()
    mean_adr_2025 = df[col_adr_2025].mean()
    mean_adr_2024 = df['ADR 2024 (SPIT)'].mean()

    delta_occ = mean_occ_2025 - mean_occ_2024
    delta_adr = mean_adr_2025 - mean_adr_2024

    if delta_occ < -10:
        occ_text = f"📉 L'occupazione è in forte calo rispetto al 2024 ({abs(delta_occ):.1f} punti % in meno)."
    else:
        occ_text = "📊 L'occupazione è relativamente stabile rispetto al 2024."

    if delta_adr < 0:
        adr_text = f"💸 L'ADR medio è sceso di circa €{abs(delta_adr):.2f} rispetto al 2024."
    else:
        adr_text = f"💰 L'ADR medio è aumentato di circa €{delta_adr:.2f} rispetto al 2024."

    st.markdown(f"**{occ_text}**\n\n**{adr_text}**")

    st.markdown("""
    ### 🔍 Raccomandazioni
    BEL CASINO
    """)

else:
    st.info("Carica un file per iniziare l'analisi.")
