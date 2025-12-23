import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import io

st.set_page_config(
    page_title="Ca' di Dio Forecast - ML Autopilot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #366092;}
    h2 {color: #366092; border-bottom: 2px solid #FFC000; padding-bottom: 10px;}
    .highlight-box {background-color: #FFF9E6; border-left: 5px solid #FFC000; padding: 15px; border-radius: 5px;}
    .autopilot-box {background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Ca' di Dio Forecast - ML Autopilot")
st.markdown("**Pickup Method con Machine Learning** | Ottimizzazione Automatica Pesi")
st.markdown("---")

# ============================================================================
# FUNZIONI ML
# ============================================================================

def calculate_mape(actual, forecast):
    """Calcola Mean Absolute Percentage Error"""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100

def optimize_weights_grid_search(baseline, year_prev, otb, actual_rn, actual_adr):
    """Ottimizza i pesi usando grid search per minimizzare MAPE"""
    
    best_mape_rn = float('inf')
    best_mape_adr = float('inf')
    best_weights = None
    
    # Grid search: prova combinazioni di pesi
    # Per velocità, usiamo step di 5%
    for w1 in range(20, 51, 5):  # Baseline: 20-50%
        for w2 in range(15, 41, 5):  # Year: 15-40%
            for w3 in range(15, 41, 5):  # OTB: 15-40%
                w4 = 100 - w1 - w2 - w3  # Pickup
                
                if w4 < 5 or w4 > 25:  # Pickup deve essere 5-25%
                    continue
                
                weights = {
                    'baseline': w1 / 100,
                    'year': w2 / 100,
                    'otb': w3 / 100,
                    'pickup': w4 / 100
                }
                
                # Forecast RN senza Biennale (lo applichiamo dopo)
                forecast_rn = (
                    baseline['rn'] * weights['baseline'] +
                    year_prev['rn'] * weights['year'] +
                    otb['rn'] * weights['otb']
                )
                
                forecast_adr = (
                    baseline['adr'] * weights['baseline'] +
                    year_prev['adr'] * weights['year'] +
                    otb['adr'] * weights['otb']
                )
                
                mape_rn = calculate_mape(actual_rn, forecast_rn)
                mape_adr = calculate_mape(actual_adr, forecast_adr)
                
                # Metrica combinata (peso uguale a RN e ADR)
                combined_mape = (mape_rn + mape_adr) / 2
                
                if combined_mape < (best_mape_rn + best_mape_adr) / 2:
                    best_mape_rn = mape_rn
                    best_mape_adr = mape_adr
                    best_weights = weights
    
    return best_weights, best_mape_rn, best_mape_adr

def calculate_forecast_simple(baseline, year_prev, otb, pickup, weights, biennale_adj, num_rooms):
    """Forecast base a 4 componenti"""
    
    rn = (
        baseline['rn'] * weights['baseline'] +
        year_prev['rn'] * weights['year'] +
        otb['rn'] * weights['otb'] +
        pickup['rn'] * weights['pickup']
    ) * biennale_adj
    
    adr = (
        baseline['adr'] * weights['baseline'] +
        year_prev['adr'] * weights['year'] +
        otb['adr'] * weights['otb'] +
        pickup['adr'] * weights['pickup']
    ) * biennale_adj
    
    return {
        'rn': rn,
        'adr': adr,
        'revenue': rn * adr,
        'occ': rn / num_rooms
    }

def identify_file_type(file):
    """Identifica il tipo di file in modo flessibile"""
    name = file.name.lower()
    
    # Baseline 2023-2024 (Biennale Arte)
    baseline_patterns = [
        'baseline', '2023-24', '2023-2024', '202324', 
        '2023_24', '2023.24', '150120', 'biennale', 'arte'
    ]
    if any(pattern in name for pattern in baseline_patterns):
        return 'baseline_2324'
    
    # Year 2024-2025 (Inflazione)
    year_patterns = [
        'year', '2024-25', '2024-2025', '202425', 
        '2024_25', '2024.25', '150108', 'inflazione', 'inflation'
    ]
    if any(pattern in name for pattern in year_patterns):
        return 'year_2425'
    
    # OTB 2026
    otb_patterns = [
        'otb', '2026', '145405', 'on the books', 
        'onthebooks', 'prenotazioni'
    ]
    if any(pattern in name for pattern in otb_patterns):
        return 'otb_2026'
    
    # Pickup RN (roomnights) - File con vs 7gg
    # Cerca prima pattern specifici per RN
    if ('145722' in name or 
        ('pickup' in name and ('rn' in name or 'roomnight' in name))):
        return 'pickup_rn'
    
    # Pickup ADR - File con ADR Room  
    # Cerca prima pattern specifici per ADR
    if ('124705' in name or 
        ('pickup' in name and 'adr' in name)):
        return 'pickup_adr'
    
    # Pickup generico - File unificato o da determinare dal contenuto
    pickup_patterns = [
        'pickup', '7gg', '7 gg', 'seven', 'booking velocity', 'velocity', 'unified'
    ]
    if any(pattern in name for pattern in pickup_patterns):
        return 'pickup_generic'
    
    # Budget
    budget_patterns = [
        'budget', 'bdg', '105652', 'budgeted', 
        'performance', 'target'
    ]
    if any(pattern in name for pattern in budget_patterns):
        return 'budget'
    
    return None

@st.cache_data
def load_data_from_uploads(files_dict):
    data = {}
    try:
        for key in ['baseline_2324', 'year_2425', 'otb_2026']:
            df = pd.read_excel(files_dict[key])
            data[key] = df[df['Giorno'].str.contains('/', na=False) & 
                          ~df['Giorno'].str.contains('Filtri', na=False)].copy()
        
        # Gestione Pickup - supporta sia file unificato che separati
        if 'pickup_generic' in files_dict:
            # File unificato - usa direttamente
            df = pd.read_excel(files_dict['pickup_generic'])
            data['pickup'] = df[df['Soggiorno'].notna()].copy()
            
        elif 'pickup_rn' in files_dict and 'pickup_adr' in files_dict:
            # File separati - merge automatico
            df_rn = pd.read_excel(files_dict['pickup_rn'])
            df_adr = pd.read_excel(files_dict['pickup_adr'])
            
            # Pulisci dataframe
            df_rn = df_rn[df_rn['Soggiorno'].notna()].copy()
            df_adr = df_adr[df_adr['Soggiorno'].notna()].copy()
            
            # Seleziona colonne rilevanti
            df_rn_clean = df_rn[['Soggiorno', 'vs 7gg']].copy()
            df_adr_clean = df_adr[['Soggiorno', 'ADR Room']].copy()
            
            # Merge automatico
            df_merged = pd.merge(df_rn_clean, df_adr_clean, on='Soggiorno', how='inner')
            data['pickup'] = df_merged
            
            st.sidebar.success("🔄 Pickup RN + ADR uniti automaticamente!")
        
        # Budget
        data['budget'] = pd.read_excel(files_dict['budget'])
        
        return data
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return None

def calc_pickup_adr(df_slice):
    pos = df_slice[df_slice['vs 7gg'] > 0]
    if len(pos) > 0 and pos['vs 7gg'].sum() > 0:
        return (pos['ADR Room'] * pos['vs 7gg']).sum() / pos['vs 7gg'].sum()
    return df_slice['ADR Room'].mean()

# ============================================================================
# SIDEBAR - FILE UPLOAD
# ============================================================================

st.sidebar.header("📁 Carica Dati")
st.sidebar.markdown("Carica **5 file** (con pickup unificato) oppure **6 file** (pickup RN e ADR separati)")
uploaded_files = st.sidebar.file_uploader("Seleziona file Excel", type=['xlsx'], accept_multiple_files=True)

files_dict = {}
if uploaded_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader("File Identificati:")
    
    for file in uploaded_files:
        file_type = identify_file_type(file)
        if file_type:
            files_dict[file_type] = file
            labels = {
                'baseline_2324': "Baseline 2023-24",
                'year_2425': "Year 2024-25",
                'otb_2026': "OTB 2026",
                'pickup_rn': "Pickup RN (roomnights)",
                'pickup_adr': "Pickup ADR (prezzi)",
                'pickup_generic': "Pickup 7gg (unificato)",
                'budget': "Budget"
            }
            label = labels.get(file_type, file_type)
            st.sidebar.markdown(f"✅ **{label}**")
        else:
            st.sidebar.warning(f"⚠️ Non riconosciuto: {file.name}")
    
    # Check required files - supporta sia 5 che 6 file
    base_required = ['baseline_2324', 'year_2425', 'otb_2026', 'budget']
    base_missing = [f for f in base_required if f not in files_dict]
    
    # Per pickup: accetta O file unificato O entrambi i file separati
    has_unified_pickup = 'pickup_generic' in files_dict
    has_separate_pickups = 'pickup_rn' in files_dict and 'pickup_adr' in files_dict
    has_pickup = has_unified_pickup or has_separate_pickups
    
    if base_missing or not has_pickup:
        missing_msg = []
        if base_missing:
            missing_msg.extend(base_missing)
        if not has_pickup:
            missing_msg.append("pickup (RN+ADR unificato) O (pickup_rn E pickup_adr)")
        
        st.sidebar.error(f"⚠️ Mancano: {', '.join(missing_msg)}")
        st.warning("⚠️ Carica tutti i file necessari")
        st.info("""
        **File necessari:**
        1. Baseline 2023-24
        2. Year 2024-25
        3. OTB 2026
        4. Budget
        5. **Pickup:** UNA di queste opzioni:
           - File unificato (con vs 7gg E ADR Room)
           - File RN (con vs 7gg) + File ADR (con ADR Room) separati
        """)
        st.stop()
    else:
        total_files = len(files_dict)
        st.sidebar.success(f"✅ Tutti i file caricati! ({total_files} file)")
        
        # Mostra info su pickup
        if has_unified_pickup:
            st.sidebar.info("📊 Pickup: File unificato rilevato")
        elif has_separate_pickups:
            st.sidebar.info("📊 Pickup: File RN + ADR separati (merge automatico)")
else:
    st.warning("⚠️ Carica i file Excel necessari")
    st.info("""
    **Opzione A (5 file):**
    1. 2023-24.xlsx (Baseline)
    2. 2024-25.xlsx (Year)
    3. otb.xlsx (OTB)
    4. pickup.xlsx (unificato con vs 7gg + ADR Room)
    5. budget.xlsx
    
    **Opzione B (6 file):**
    1. 2023-24.xlsx (Baseline)
    2. 2024-25.xlsx (Year)
    3. otb.xlsx (OTB)
    4. pickup_rn.xlsx (con vs 7gg)
    5. pickup_adr.xlsx (con ADR Room)
    6. budget.xlsx
    """)
    st.stop()

data = load_data_from_uploads(files_dict)
if data is None:
    st.stop()

# Verifica che pickup abbia le colonne necessarie
required_pickup_cols = ['ADR Room', 'vs 7gg']
missing_cols = [col for col in required_pickup_cols if col not in data['pickup'].columns]

if missing_cols:
    st.error(f"❌ File Pickup mancano colonne: {', '.join(missing_cols)}")
    st.info("""
    **Il file pickup deve avere:**
    - Colonna `ADR Room` (ADR medio prenotazioni)
    - Colonna `vs 7gg` (roomnights pickup)
    
    **Soluzioni:**
    1. Usa file unificato con entrambe le colonne
    2. Oppure carica 2 file separati (RN + ADR) che verranno uniti automaticamente
    """)
    st.stop()
else:
    st.sidebar.success("✅ File Pickup validato correttamente!")

st.sidebar.markdown("---")

# ============================================================================
# SIDEBAR - DATA DINAMICA
# ============================================================================

st.sidebar.header("📅 Data Riferimento")

# Estrai data dal nome file o usa oggi
default_date = datetime.now()
for file in uploaded_files:
    if 'otb' in file.name.lower():
        # Cerca pattern data nel filename (es: 2025-12-16)
        import re
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', file.name)
        if match:
            default_date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            break

report_date = st.sidebar.date_input(
    "Data Report OTB",
    value=default_date,
    help="Data in cui è stato scaricato l'OTB - determina lo split Actual/Forecast"
)

report_date = datetime.combine(report_date, datetime.min.time())

# Calcola split dinamico per Dicembre
giorni_actual_dic = report_date.day if report_date.month == 12 else 0
giorni_forecast_dic = 31 - giorni_actual_dic if report_date.month == 12 else 31

st.sidebar.info(f"""
**Split Dicembre 2025:**
- Actual: 1-{giorni_actual_dic} ({giorni_actual_dic}gg)
- Forecast: {giorni_actual_dic+1}-31 ({giorni_forecast_dic}gg)
""")

st.sidebar.markdown("---")

# ============================================================================
# SIDEBAR - MODALITÀ
# ============================================================================

st.sidebar.header("🤖 Modalità Forecast")

mode = st.sidebar.radio(
    "Seleziona modalità:",
    ["Manual", "Autopilot ML"],
    help="Manual: imposti i pesi manualmente | Autopilot: ML ottimizza automaticamente"
)

if mode == "Manual":
    st.sidebar.subheader("Pesi Manuali")
    peso_baseline = st.sidebar.slider("① Baseline 2024", 0, 100, 35, 5)
    peso_year = st.sidebar.slider("② Anno 2025", 0, 100, 25, 5)
    peso_otb = st.sidebar.slider("③ OTB 2026", 0, 100, 25, 5)
    peso_pickup = st.sidebar.slider("④ Pickup 7gg", 0, 100, 15, 5)
    
    peso_totale = peso_baseline + peso_year + peso_otb + peso_pickup
    if peso_totale != 100:
        st.sidebar.error(f"⚠️ TOTALE: {peso_totale}%")
    else:
        st.sidebar.success(f"✅ TOTALE: {peso_totale}%")
    
    weights = {
        'baseline': peso_baseline / 100,
        'year': peso_year / 100,
        'otb': peso_otb / 100,
        'pickup': peso_pickup / 100
    }
    
    ml_used = False
    
else:  # Autopilot
    st.sidebar.markdown("""
    <div class="autopilot-box">
    <strong>🤖 Autopilot Attivo</strong><br>
    Il sistema ottimizzerà automaticamente i pesi per minimizzare l'errore MAPE.
    </div>
    """, unsafe_allow_html=True)
    
    # Usa dati storici per ottimizzare
    # Per semplicità, uso baseline 2024 come "actual" e ottimizzo i pesi
    
    with st.spinner("🔄 Ottimizzazione ML in corso..."):
        # Prepara dati per ottimizzazione (usa Gen 2024 come test)
        test_baseline = {
            'rn': data['baseline_2324'].iloc[32:63]['Room nights'].sum(),
            'adr': data['baseline_2324'].iloc[32:63]['ADR Cam'].mean()
        }
        test_year = {
            'rn': data['year_2425'].iloc[31:62]['Room nights'].sum(),
            'adr': data['year_2425'].iloc[31:62]['ADR Cam'].mean()
        }
        test_otb = {
            'rn': data['otb_2026'].iloc[31:62]['Room nights'].sum(),
            'adr': data['otb_2026'].iloc[31:62]['ADR Cam'].mean()
        }
        
        # Target: usa baseline come "actual"
        actual_rn = test_baseline['rn']
        actual_adr = test_baseline['adr']
        
        # Ottimizza
        best_weights, mape_rn, mape_adr = optimize_weights_grid_search(
            test_baseline, test_year, test_otb, actual_rn, actual_adr
        )
        
        weights = best_weights
        ml_used = True
        
        st.sidebar.success(f"""
        ✅ **Pesi Ottimizzati:**
        - Baseline: {weights['baseline']:.0%}
        - Year: {weights['year']:.0%}
        - OTB: {weights['otb']:.0%}
        - Pickup: {weights['pickup']:.0%}
        
        **Performance:**
        - MAPE RN: {mape_rn:.2f}%
        - MAPE ADR: {mape_adr:.2f}%
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("Biennale Adjustment")
biennale_adj = st.sidebar.number_input("Fattore", 1.00, 1.50, 1.10, 0.01)

# ============================================================================
# CALCOLI FORECAST CON DATA DINAMICA
# ============================================================================

try:
    num_rooms = 66
    
    # DICEMBRE - con split dinamico
    if giorni_actual_dic > 0:
        dic_actual = {
            'rn': data['otb_2026'].iloc[0:giorni_actual_dic]['Room nights'].sum(),
            'adr': data['otb_2026'].iloc[0:giorni_actual_dic]['ADR Cam'].mean(),
            'revenue': data['otb_2026'].iloc[0:giorni_actual_dic]['Room Revenue'].sum(),
        }
        dic_actual['occ'] = dic_actual['rn'] / (num_rooms * giorni_actual_dic)
    else:
        dic_actual = {'rn': 0, 'adr': 0, 'revenue': 0, 'occ': 0}
    
    # Forecast per giorni rimanenti
    if giorni_forecast_dic > 0:
        start_idx = giorni_actual_dic
        end_idx = 31
        
        dic_fcst_baseline = {
            'rn': data['baseline_2324'].iloc[start_idx:end_idx]['Room nights'].sum(),
            'adr': data['baseline_2324'].iloc[start_idx:end_idx]['ADR Cam'].mean()
        }
        dic_fcst_year = {
            'rn': data['year_2425'].iloc[start_idx:end_idx]['Room nights'].sum(),
            'adr': data['year_2425'].iloc[start_idx:end_idx]['ADR Cam'].mean()
        }
        dic_fcst_otb = {
            'rn': data['otb_2026'].iloc[start_idx:end_idx]['Room nights'].sum(),
            'adr': data['otb_2026'].iloc[start_idx:end_idx]['ADR Cam'].mean()
        }
        dic_fcst_pickup = {
            'rn': data['pickup'].iloc[start_idx:end_idx]['vs 7gg'].sum(),
            'adr': calc_pickup_adr(data['pickup'].iloc[start_idx:end_idx])
        }
        
        dic_fcst = calculate_forecast_simple(
            dic_fcst_baseline, dic_fcst_year, dic_fcst_otb, dic_fcst_pickup,
            weights, biennale_adj, num_rooms * giorni_forecast_dic
        )
    else:
        dic_fcst = {'rn': 0, 'adr': 0, 'revenue': 0, 'occ': 0}
    
    dic_total = {
        'rn': dic_actual['rn'] + dic_fcst['rn'],
        'revenue': dic_actual['revenue'] + dic_fcst['revenue'],
    }
    dic_total['adr'] = dic_total['revenue'] / dic_total['rn'] if dic_total['rn'] > 0 else 0
    dic_total['occ'] = dic_total['rn'] / (num_rooms * 31)
    
    # GENNAIO
    gen_baseline = {
        'rn': data['baseline_2324'].iloc[32:63]['Room nights'].sum(),
        'adr': data['baseline_2324'].iloc[32:63]['ADR Cam'].mean()
    }
    gen_year = {
        'rn': data['year_2425'].iloc[31:62]['Room nights'].sum(),
        'adr': data['year_2425'].iloc[31:62]['ADR Cam'].mean()
    }
    gen_otb = {
        'rn': data['otb_2026'].iloc[31:62]['Room nights'].sum(),
        'adr': data['otb_2026'].iloc[31:62]['ADR Cam'].mean()
    }
    gen_pickup = {
        'rn': data['pickup'].iloc[31:62]['vs 7gg'].sum(),
        'adr': calc_pickup_adr(data['pickup'].iloc[31:62])
    }
    
    gen_fcst = calculate_forecast_simple(
        gen_baseline, gen_year, gen_otb, gen_pickup,
        weights, biennale_adj, num_rooms * 31
    )
    
    # FEBBRAIO
    feb_baseline = {
        'rn': data['baseline_2324'].iloc[63:92]['Room nights'].sum(),
        'adr': data['baseline_2324'].iloc[63:92]['ADR Cam'].mean()
    }
    feb_year = {
        'rn': data['year_2425'].iloc[62:91]['Room nights'].sum(),
        'adr': data['year_2425'].iloc[62:91]['ADR Cam'].mean()
    }
    feb_otb = {
        'rn': data['otb_2026'].iloc[62:90]['Room nights'].sum(),
        'adr': data['otb_2026'].iloc[62:90]['ADR Cam'].mean()
    }
    feb_pickup = {
        'rn': data['pickup'].iloc[62:90]['vs 7gg'].sum(),
        'adr': calc_pickup_adr(data['pickup'].iloc[62:90])
    }
    
    feb_fcst = calculate_forecast_simple(
        feb_baseline, feb_year, feb_otb, feb_pickup,
        weights, biennale_adj, num_rooms * 28
    )
    
    # Budget
    budget_data = {
        'dic': {
            'rn': data['budget'].iloc[1]['Roomnights BDG'],
            'adr': data['budget'].iloc[1]['ADR Room BDG'],
            'revenue': data['budget'].iloc[1]['Room Revenue BDG'],
            'occ': data['budget'].iloc[1]['Occ.% BDG']
        },
        'gen': {
            'rn': data['budget'].iloc[2]['Roomnights BDG'],
            'adr': data['budget'].iloc[2]['ADR Room BDG'],
            'revenue': data['budget'].iloc[2]['Room Revenue BDG'],
            'occ': data['budget'].iloc[2]['Occ.% BDG']
        },
        'feb': {
            'rn': data['budget'].iloc[3]['Roomnights BDG'],
            'adr': data['budget'].iloc[3]['ADR Room BDG'],
            'revenue': data['budget'].iloc[3]['Room Revenue BDG'],
            'occ': data['budget'].iloc[3]['Occ.% BDG']
        }
    }

except Exception as e:
    st.error(f"❌ Errore: {e}")
    st.stop()

# ============================================================================
# DASHBOARD
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 ML Insights", "📈 Grafici", "💾 Export"])

with tab1:
    st.header("Dashboard KPI")
    
    if ml_used:
        st.markdown(f"""
        <div class="autopilot-box">
        <strong>🤖 Modalità Autopilot Attiva</strong><br>
        Pesi ottimizzati: Baseline {weights['baseline']:.0%} | Year {weights['year']:.0%} | OTB {weights['otb']:.0%} | Pickup {weights['pickup']:.0%}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="highlight-box">
    <strong>Split Dicembre:</strong> Actual 1-{giorni_actual_dic} | Forecast {giorni_actual_dic+1}-31<br>
    <strong>Data Report:</strong> {report_date.strftime('%d/%m/%Y')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # DICEMBRE
    st.subheader("🎄 DICEMBRE 2025")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Roomnights", f"{dic_total['rn']:,.0f}",
                 f"{((dic_total['rn']/budget_data['dic']['rn']-1)*100):.1f}% vs BDG")
    with col2:
        st.metric("ADR Camera", f"€{dic_total['adr']:,.2f}",
                 f"{((dic_total['adr']/budget_data['dic']['adr']-1)*100):.1f}% vs BDG")
    with col3:
        st.metric("Revenue", f"€{dic_total['revenue']:,.0f}",
                 f"{((dic_total['revenue']/budget_data['dic']['revenue']-1)*100):.1f}% vs BDG")
    with col4:
        st.metric("Occupancy", f"{dic_total['occ']:.1%}",
                 f"{(dic_total['occ']-budget_data['dic']['occ']):.1%} vs BDG")
    
    st.markdown("---")
    
    # GENNAIO
    st.subheader("❄️ GENNAIO 2026")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Roomnights", f"{gen_fcst['rn']:,.0f}",
                 f"{((gen_fcst['rn']/budget_data['gen']['rn']-1)*100):.1f}% vs BDG")
    with col2:
        st.metric("ADR Camera", f"€{gen_fcst['adr']:,.2f}",
                 f"{((gen_fcst['adr']/budget_data['gen']['adr']-1)*100):.1f}% vs BDG")
    with col3:
        st.metric("Revenue", f"€{gen_fcst['revenue']:,.0f}",
                 f"{((gen_fcst['revenue']/budget_data['gen']['revenue']-1)*100):.1f}% vs BDG")
    with col4:
        st.metric("Occupancy", f"{gen_fcst['occ']:.1%}",
                 f"{(gen_fcst['occ']-budget_data['gen']['occ']):.1%} vs BDG")
    
    st.markdown("---")
    
    # FEBBRAIO
    st.subheader("💝 FEBBRAIO 2026")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Roomnights", f"{feb_fcst['rn']:,.0f}",
                 f"{((feb_fcst['rn']/budget_data['feb']['rn']-1)*100):.1f}% vs BDG")
    with col2:
        st.metric("ADR Camera", f"€{feb_fcst['adr']:,.2f}",
                 f"{((feb_fcst['adr']/budget_data['feb']['adr']-1)*100):.1f}% vs BDG")
    with col3:
        st.metric("Revenue", f"€{feb_fcst['revenue']:,.0f}",
                 f"{((feb_fcst['revenue']/budget_data['feb']['revenue']-1)*100):.1f}% vs BDG")
    with col4:
        st.metric("Occupancy", f"{feb_fcst['occ']:.1%}",
                 f"{(feb_fcst['occ']-budget_data['feb']['occ']):.1%} vs BDG")

with tab2:
    st.header("🤖 Machine Learning Insights")
    
    if ml_used:
        st.success("✅ Autopilot attivo - Pesi ottimizzati automaticamente")
        
        st.subheader("Pesi Ottimizzati")
        
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            x=['Baseline 2024', 'Anno 2025', 'OTB 2026', 'Pickup 7gg'],
            y=[weights['baseline']*100, weights['year']*100, weights['otb']*100, weights['pickup']*100],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
            text=[f"{weights['baseline']:.0%}", f"{weights['year']:.0%}", 
                  f"{weights['otb']:.0%}", f"{weights['pickup']:.0%}"],
            textposition='auto'
        ))
        fig_weights.update_layout(
            title='Distribuzione Pesi Ottimizzati',
            yaxis_title='Peso (%)',
            height=400
        )
        st.plotly_chart(fig_weights, use_container_width=True)
        
        st.info(f"""
        **Performance del Modello:**
        - MAPE Roomnights: {mape_rn:.2f}%
        - MAPE ADR: {mape_adr:.2f}%
        - Combined MAPE: {(mape_rn + mape_adr)/2:.2f}%
        
        *MAPE < 10%: Eccellente | 10-20%: Buono | >20%: Migliorabile*
        """)
        
    else:
        st.info("ℹ️ Modalità Manual attiva - Passa ad Autopilot per ottimizzazione ML")
        
        st.markdown("""
        **Modalità Autopilot disponibili:**
        
        1. **Grid Search** ✅ (Attuale)
           - Testa sistematicamente combinazioni di pesi
           - Minimizza MAPE su dati storici
           - Veloce e affidabile
        
        2. **Random Forest** (Coming Soon)
           - ML ensemble method
           - Cattura relazioni non-lineari
        
        3. **Gradient Boosting** (Coming Soon)
           - Ottimizzazione avanzata
           - Performance superiore su pattern complessi
        
        4. **Prophet** (Coming Soon)
           - Time series forecasting di Facebook
           - Gestione automatica stagionalità
        """)

with tab3:
    st.header("📈 Grafici Comparativi")
    
    months = ['Dicembre', 'Gennaio', 'Febbraio']
    fcst_rev = [dic_total['revenue'], gen_fcst['revenue'], feb_fcst['revenue']]
    bdg_rev = [budget_data['dic']['revenue'], budget_data['gen']['revenue'], budget_data['feb']['revenue']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Forecast', x=months, y=fcst_rev, marker_color='#366092'))
    fig.add_trace(go.Bar(name='Budget', x=months, y=bdg_rev, marker_color='#FFC000'))
    fig.update_layout(
        title='Revenue: Forecast vs Budget',
        yaxis_title='Revenue (€)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("💾 Export Risultati")
    
    export_df = pd.DataFrame({
        'Mese': ['Dicembre', 'Gennaio', 'Febbraio'],
        'Forecast RN': [dic_total['rn'], gen_fcst['rn'], feb_fcst['rn']],
        'Forecast ADR': [dic_total['adr'], gen_fcst['adr'], feb_fcst['adr']],
        'Forecast Revenue': [dic_total['revenue'], gen_fcst['revenue'], feb_fcst['revenue']],
        'Modalità': ['Autopilot ML' if ml_used else 'Manual'] * 3,
        'Data Report': [report_date.strftime('%d/%m/%Y')] * 3
    })
    
    st.dataframe(export_df, use_container_width=True, hide_index=True)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False)
    
    st.download_button(
        "📥 Scarica Excel",
        output.getvalue(),
        f"Forecast_ML_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <strong>Ca' di Dio Vretreats</strong> - ML Autopilot 🤖 | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
