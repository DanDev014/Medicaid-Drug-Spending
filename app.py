import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Loading the trained model
model = joblib.load("medicaid_spending_model.pkl")

# State abbreviation mapping 
state_full_to_abbr = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY", "District of Columbia": "DC", "Puerto Rico": "PR"
}

# Full list of 683 unique cleaned drug names 
all_drugs = [
    '0.9% sodiu', 'abilify as', 'abilify ma', 'acetaminop', 'actemra ac', 'actemra pe', 'actimmune', 'adbry auto',
    'adcetris', 'adderall x', 'adempas', 'adempas 2.', 'advair dis', 'advair hfa', 'advate 5ml', 'agamree',
    'agamree ki', 'aimovig (e', 'aimovig au', 'ajovy (fre', 'ajovy auto', 'albuterol', 'alogliptin', 'alprazolam',
    'altuviiio', 'alyftrek', 'alyftrek (', 'amcinonide', 'amlodipine', 'amondys 45', 'amondys-45', 'amoxicilli',
    'amphetamin', 'anoro elli', 'apretude', 'aptiom', 'aptiom 800', 'aripiprazo', 'aristada', 'aristada 1',
    'aristada 8', 'aristada e', 'arnuity el', 'atomoxetin', 'atorvastat', 'atrovent h', 'austedo', 'austedo (t',
    'austedo t', 'austedo xr', 'auvelity', 'azelastine', 'azithromyc', 'azstarys', 'banzel', 'baqsimi', 'basaglar',
    'benlysta', 'benlysta l', 'benlysta s', 'benzonatat', 'bicillin l', 'biktarvy', 'biktarvy 5', 'biktarvy b',
    'bimzelx 1', 'bimzelx 2', 'bimzelx au', 'blincyto', 'blincyto (', 'botox', 'botox botu', 'breo ellip',
    'breztri 16', 'breztri ae', 'bridion', 'brilinta', 'brilinta 9', 'briviact', 'briviact (', 'brixadi',
    'brukinsa', 'budesonide', 'bupren&nal', 'buprenorph', 'bupropion', 'buspirone', 'cabenuva', 'cabenuva i',
    'cabometyx', 'calquence', 'caplyta', 'caplyta 42', 'caplyta ca', 'cefazolin', 'ceftriaxon', 'cephalexin',
    'cetirizine', 'cimzia (2', 'cimzia pre', 'ciprofloxa', 'clindamyci', 'clonazepam', 'clonidine', 'clotrimazo',
    'clozapine', 'combivent', 'concerta', 'concerta 1', 'concerta 2', 'concerta 3', 'concerta 5', 'concerta e',
    'cosentyx s', 'cosentyx u', 'creon', 'creon (pan', 'creon 2400', 'creon dr 3', 'creon pan', 'crysvita',
    'crysvita (', 'cyclobenza', 'dapagliflo', 'darzalex f', 'dasatinib', 'daybue', 'daybue 200', 'daybue ora',
    'descovy', 'descovy 20', 'dexamethas', 'dexmethylp', 'dextroamph', 'diclofenac', 'diphenhydr', 'diprivan',
    'divalproex', 'dolobid', 'dovato', 'dovato 50-', 'dovato tab', 'doxepin hc', 'doxycyclin', 'dulera', 'dulera 100',
    'dulera 200', 'duloxetine', 'dupixent 2', 'dupixent 3', 'dupixent p', 'dupixent s', 'duvyzat', 'duvyzat or',
    'elaprase', 'elaprase 6', 'eliquis', 'eliquis 5', 'emflaza', 'emflaza (d', 'emgality', 'emgality p', 'enbrel',
    'enbrel (et', 'enbrel 50', 'enbrel et', 'enbrel sur', 'enhertu', 'enhertu-10', 'enoxaparin', 'entresto',
    'entresto 2', 'entresto f', 'entyvio', 'entyvio 30', 'envarsus x', 'epclusa 40', 'epidiolex', 'epinephrin',
    'epogen', 'erbitux', 'ergocalcif', 'erythromyc', 'escitalopr', 'estradiol', 'eucrisa', 'evrysdi', 'evrysdi (r',
    'evrysdi 60', 'evrysdi r', 'exondys 51', 'exondys-51', 'eylea', 'eylea (afl', 'eylea 2 mg', 'eylea afl',
    'eylea hd 8', 'fabrazyme', 'famotidine', 'farxiga', 'farxiga 10', 'farxiga 5m', 'fasenra pe', 'fentanyl c',
    'ferriprox', 'filsuvez 1', 'fintepla', 'fintepla 2', 'fluconazol', 'fluoxetine', 'fluticason', 'focalin xr',
    'folic acid', 'fycompa', 'gabapentin', 'gammagard', 'gammaplex', 'gamunex-c', 'gattex', 'gattex - 3',
    'gattex 5 m', 'gemtesa', 'genotropin', 'genvoya', 'genvoya ta', 'guanfacine', 'gvoke hypo', 'hadlima 40',
    'hemlibra', 'hemlibra -', 'hemlibra 1', 'hemlibra 6', 'heparin 30', 'heparin so', 'hizentra', 'hizentra 4',
    'hizentra p', 'humalog', 'humalog kw', 'humira', 'humira - a', 'humira 80m', 'humira pen', 'humira pfs',
    'humira(cf)', 'humulin r', 'hydrocodon', 'hydrocorti', 'hydroxym', 'hydroxyzin', 'ibrance', 'ibrance 12',
    'ibsrela', 'ibuprofen', 'idelvion', 'ilaris', 'ilaris via', 'imbruvica', 'imcivree', 'imcivree (', 'imfinzi',
    'imfinzi, 5', 'incruse el', 'inflectra', 'infliximab', 'ingrezza', 'ingrezza 4', 'ingrezza 6', 'ingrezza 8',
    'injectafer', 'insulin as', 'insulin gl', 'insulin li', 'invega haf', 'invega sus', 'invega tri', 'invokana',
    'ipratropiu', 'isentress', 'isotretino', 'jakafi', 'janumet', 'janumet 50', 'januvia', 'januvia 10',
    'januvia 50', 'jardiance', 'jentadueto', 'jornay pm', 'jublia', 'juluca', 'juluca 50m', 'jynarque', 'jynarque 4',
    'kerendia 1', 'kesimpta 2', 'kesimpta p', 'kesimpta s', 'ketoconazo', 'ketorolac', 'keytruda', 'keytruda 1',
    'kisqali', 'kisqali 60', 'kisqali fc', 'koselugo', 'krystexxa', 'kyleena', 'lacosamide', 'lactated r',
    'lamotrigin', 'lantus 3ml', 'lantus sol', 'lenalidomi', 'lenvima', 'levetirace', 'levothyrox', 'lidocaine',
    'linzess', 'linzess 14', 'linzess 29', 'linzess 72', 'liraglutid', 'lisdexamfe', 'lisinopril', 'livmarli',
    'lo loestri', 'loratadine', 'losartan p', 'lupron dep', 'lurasidone', 'lybalvi', 'lybalvi (1', 'lynparza',
    'lynparza 1', 'magnesium', 'mavyret', 'mavyret 10', 'medroxypro', 'mekinist', 'mekinist t', 'meloxicam',
    'mesalamine', 'metformin', 'methylphen', 'methylpred', 'metoprolol', 'metronidaz', 'midazolam', 'mirabegron',
    'mirena', 'mirtazapin', 'mixed amph', 'montelukas', 'morphine s', 'mounjaro', 'mounjaro 1', 'mounjaro 2',
    'mounjaro 5', 'mounjaro 7', 'mupirocin', 'mycophenol', 'myfembree', 'myrbetriq', 'naloxone h', 'naltrexone',
    'nayzilam', 'nayzilam 5', 'nemluvio', 'nexplanon', 'nicotine t', 'nitrofuran', 'norditropi', 'novolog fl',
    'novoseven', 'nubeqa', 'nucala', 'nucala sol', 'nuedexta', 'nurtec odt', 'nuwiq', 'ocrevus', 'ocrevus 30',
    'octagam 10', 'odefsey', 'odefsey (e', 'odefsey e', 'ofev', 'ofev (nint', 'ohtuvayre', 'olanzapine',
    'omeprazole', 'omnipaque', 'omnitrope', 'ondansetro', 'onfi', 'opdivo', 'opsumit', 'opsumit (m', 'opsumit 10',
    'opsumit m', 'opzelura', 'opzelura 1', 'orencia', 'orencia cl', 'orencia sc', 'orenitram', 'oseltamivi',
    'otezla', 'otezla (ap', 'otezla 30', 'otezla ap', 'oxcarbazep', 'oxervate', 'oxtellar x', 'oxycodone',
    'oxycontin', 'ozempic', 'ozempic 0.', 'ozempic 1', 'ozempic 1m', 'ozempic 2', 'ozempic 2m', 'pantoprazo',
    'paragard t', 'paxlovid', 'paxlovid 1', 'paxlovid 3', 'perjeta', 'perjeta (p', 'polyethyle', 'pomalyst',
    'potassium', 'prednisolo', 'prednisone', 'pregabalin', 'prezcobix', 'privigen', 'prolia (de', 'promacta',
    'promacta t', 'promethazi', 'propranolo', 'pulmicort', 'pulmozyme', 'qelbree', 'qelbree 20', 'quetiapine',
    'quillichew', 'quillivant', 'qulipta', 'qulipta (a', 'qvar ba in', 'qvar redih', 'ravicti', 'ravicti 1.',
    'relafen ds', 'relistor', 'relistor t', 'remicade', 'remodulin', 'repatha (e', 'repatha e', 'repatha su',
    'restasis', 'revlimid', 'revlimid (', 'rexulti', 'rexulti 0.', 'rexulti 1.', 'rexulti 2.', 'rexulti 3.',
    'rezdiffra', 'rezdiffra/', 'rinvoq', 'rinvoq 15m', 'rinvoq 30m', 'rinvoq 45m', 'rinvoq er', 'risperdal',
    'risperidon', 'rosuvastat', 'ruconest', 'rybelsus', 'rybelsus 1', 'rybelsus 3', 'rybelsus 7', 'sabril',
    'sabril 500', 'saphnelo', 'saxenda', 'scemblix', 'scemblix f', 'sertraline', 'simponi 50', 'simponi ar',
    'skyclarys', 'skyrizi', 'skyrizi 15', 'skyrizi 36', 'skyrizi 60', 'skyrizi in', 'skyrizi on', 'skyrizi pe',
    'skytrofa', 'slynd', 'sodium chl', 'sodium cl', 'sodium oxy', 'sofosbuvir', 'soliris', 'somatuline', 'sotyktu',
    'spiriva 30', 'spiriva ha', 'spiriva re', 'spironolac', 'spravato', 'spravato n', 'sprycel', 'steglatro',
    'stelara', 'stelara 45', 'stelara 90', 'sterile wa', 'stiolto re', 'strensiq', 'strensiq 8', 'sublocade',
    'suboxone', 'suboxone 1', 'suboxone 8', 'suboxone s', 'sucralfate', 'sumatripta', 'symbicort', 'symtuza',
    'symtuza 80', 'synjardy 1', 'synjardy x', 'synthroid', 'tacrolimus', 'tagrisso', 'tagrisso 8', 'takhzyro',
    'takhzyro 3', 'taltz', 'taltz 80 m', 'taltz auto', 'tasigna hg', 'tecentriq', 'tepezza', 'testostero',
    'tezspire', 'tezspire (', 'tiotropium', 'tivicay', 'tivicay 50', 'tivicay ta', 'topiramate', 'toujeo max',
    'toujeo sol', 'tradjenta', 'trazodone', 'trelegy el', 'tremfya', 'tremfya 1x', 'tremfya on', 'tresiba fl',
    'tretinoin', 'triamcinol', 'trikafta', 'trikafta (', 'trikafta 1', 'trikafta 5', 'trileptal', 'trintellix',
    'triumeq', 'triumeq 50', 'triumeq 60', 'trodelvy', 'trulicity', 'tysabri', 'tysabri (n', 'tyvaso dpi',
    'ubrelvy', 'ubrelvy 10', 'ubrelvy 50', 'ultomiris', 'uptravi', 'uptravi (s', 'uptravi s', 'uzedy',
    'uzedy er i', 'vabysmo', 'vabysmo 6m', 'valacyclov', 'valtoco', 'valtoco 10', 'valtoco 15', 'vareniclin',
    'velphoro', 'vemlidy', 'vemlidy 25', 'venclexta', 'venlafaxin', 'venofer', 'ventolin h', 'verzenio',
    'verzenio 1', 'vigabatrin', 'vigadrone', 'viltepso 2', 'vimizim', 'vitamin d2', 'vivitrol', 'vivitrol (',
    'vivitrol 3', 'voquezna', 'voranigo', 'vraylar', 'vraylar (c', 'vraylar 1.', 'vraylar 3', 'vraylar c',
    'vumerity 2', 'vyjuvek', 'vyvanse', 'vyvanse 10', 'vyvanse 20', 'vyvanse 30', 'vyvanse 40', 'vyvanse 50',
    'vyvanse 60', 'vyvanse 70', 'vyvanse ch', 'vyvgart hy', 'wakix', 'wakix (pit', 'wegovy', 'wegovy 0.2',
    'wegovy 0.5', 'wegovy 1.7', 'wegovy 1mg', 'wegovy 2.4', 'winrevair', 'xarelto', 'xarelto (r', 'xarelto 10',
    'xarelto 20', 'xarelto r', 'xcopri', 'xcopri cen', 'xeljanz', 'xeljanz to', 'xeljanz xr', 'xgeva (den',
    'xifaxan', 'xifaxan 55', 'xigduo xr', 'xiidra', 'xiidra sol', 'xolair', 'xolair 150', 'xolair 300', 'xtandi',
    'xtandi 40m', 'xtandi 80m', 'xywav', 'xywav (cal', 'yervoy', 'zenpep', 'zenpep cap', 'zenpep dr', 'zepbound',
    'zepbound 1', 'zepbound 2', 'zepbound 5', 'zepbound 7', 'zoryve', 'zoryve / 6', 'ztlido', 'zurzuvae 2'
]

# Full list of states 
all_states_full = sorted(state_full_to_abbr.keys())

# Spending data –
state_spending = pd.DataFrame({
    "State Full Name": [
        "California", "New York", "Pennsylvania", "Ohio", "North Carolina",
        "Michigan", "Illinois", "Florida", "Indiana", "Virginia",
        "Kentucky", "Texas", "Louisiana", "Massachusetts", "Wisconsin",
        "Arizona", "Connecticut", "New Jersey", "Missouri", "Minnesota",
        "Colorado", "Tennessee", "Washington", "Maryland", "Georgia",
        "Puerto Rico", "Alabama", "Oklahoma", "Oregon", "South Carolina",
        "Iowa", "West Virginia", "Nevada", "Idaho", "New Mexico",
        "Mississippi", "Nebraska", "Maine", "Arkansas", "Utah",
        "Kansas", "Rhode Island", "District of Columbia", "Montana",
        "South Dakota", "New Hampshire", "Delaware", "Hawaii",
        "Alaska", "Vermont", "North Dakota", "Wyoming"
    ],
    "Total Amount Reimbursed": [
        6.730810e+09, 5.143920e+09, 2.345845e+09, 2.009043e+09, 2.001101e+09,
        1.755784e+09, 1.442779e+09, 1.387497e+09, 1.267096e+09, 1.240000e+09,
        1.181363e+09, 1.163783e+09, 1.108245e+09, 9.837195e+08, 9.464532e+08,
        9.020180e+08, 8.812838e+08, 8.467704e+08, 8.321257e+08, 7.172326e+08,
        7.135693e+08, 6.708751e+08, 6.695321e+08, 6.414715e+08, 5.614090e+08,
        5.572708e+08, 4.696355e+08, 4.587712e+08, 4.440559e+08, 4.312609e+08,
        4.303864e+08, 3.431055e+08, 3.332807e+08, 2.697718e+08, 2.617272e+08,
        2.584661e+08, 2.223908e+08, 2.165565e+08, 2.151721e+08, 1.885156e+08,
        1.673107e+08, 1.398654e+08, 1.350567e+08, 1.350160e+08, 1.291709e+08,
        1.241681e+08, 1.116113e+08, 1.029105e+08, 9.982954e+07, 8.002319e+07,
        4.154311e+07, 2.386073e+07
    ]
})
state_spending["State Code"] = state_spending["State Full Name"].map(state_full_to_abbr)

# Usage data 
state_usage = pd.DataFrame({
    "State Full Name": [
        "California", "New York", "Ohio", "Pennsylvania", "Texas",
        "North Carolina", "Michigan", "Florida", "Kentucky", "Illinois",
        "Indiana", "New Jersey", "Virginia", "Missouri", "Massachusetts",
        "Minnesota", "Louisiana", "Arizona", "Washington", "Tennessee",
        "Georgia", "Maryland", "Wisconsin", "Oregon", "Colorado",
        "Connecticut", "Oklahoma", "Iowa", "Puerto Rico", "West Virginia",
        "South Carolina", "Alabama", "Nevada", "New Mexico", "Mississippi",
        "Idaho", "Arkansas", "Nebraska", "Delaware", "Maine",
        "Utah", "Rhode Island", "Kansas", "New Hampshire", "Montana",
        "District of Columbia", "Hawaii", "Alaska", "Vermont",
        "South Dakota", "North Dakota", "Wyoming"
    ],
    "Units Reimbursed": [
        2.037401e+09, 1.488311e+09, 7.663352e+08, 7.039589e+08, 6.490379e+08,
        5.959167e+08, 5.167147e+08, 4.880454e+08, 4.733878e+08, 4.474310e+08,
        3.938806e+08, 3.557094e+08, 3.473175e+08, 3.401117e+08, 3.317646e+08,
        3.054770e+08, 3.032595e+08, 2.865433e+08, 2.801513e+08, 2.751059e+08,
        2.608771e+08, 2.577577e+08, 2.486559e+08, 2.172187e+08, 2.038336e+08,
        1.888893e+08, 1.808091e+08, 1.772497e+08, 1.666528e+08, 1.460088e+08,
        1.440507e+08, 1.354221e+08, 1.282940e+08, 1.270166e+08, 1.114967e+08,
        1.107530e+08, 1.042506e+08, 1.036198e+08, 9.471249e+07, 7.424043e+07,
        7.310756e+07, 6.448514e+07, 6.283161e+07, 5.459353e+07, 4.697207e+07,
        4.249829e+07, 3.979708e+07, 3.752678e+07, 3.314500e+07, 2.504632e+07,
        2.108415e+07, 8.125119e+06
    ]
})
state_usage["State Code"] = state_usage["State Full Name"].map(state_full_to_abbr)

# Top expensive drugs
top_exp_drugs = pd.DataFrame({
    "Product Name": ["biktarvy", "jardiance", "trulicity", "invega sus", "humira(cf)",
                     "humira pen", "ozempic", "dupixent s", "eliquis", "zepbound",
                     "dupixent p", "ozempic 0.", "abilify ma", "stelara 90", "vraylar (c"],
    "Total Amount Reimbursed": [1.487752e9, 1.224935e9, 9.942157e8, 8.841717e8, 8.621136e8,
                                7.755576e8, 6.612600e8, 6.432030e8, 6.114578e8, 5.080595e8,
                                4.911089e8, 4.176484e8, 4.130071e8, 3.995239e8, 3.830521e8]
})

# Most prescribed drugs
popular_drugs = pd.DataFrame({
    "Product Name": ["amoxicilli", "albuterol", "ibuprofen", "fluticason", "atorvastat",
                     "gabapentin", "ondansetro", "cetirizine", "metformin", "sertraline",
                     "hydroxyzin", "omeprazole", "amlodipine", "lisinopril", "trazodone"],
    "No of prescriptions": [8243538, 7264826, 6543001, 6042960, 5987791,
                            5969243, 5741383, 5275403, 4655398, 4481159,
                            4438048, 4397245, 4231409, 4016472, 3957128]
})

# Page config & styling
st.set_page_config(page_title="MedicaidRx Estimator", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; }
    h1, h2, h3 { color: #003366; }
    .stButton>button { 
        background-color: #003366; 
        color: white; 
        border: none; 
        border-radius: 6px; 
        padding: 0.6em 1.4em;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e6f0ff;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #003366;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003366 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 Medicaid: Spending Estimator & Spending Patterns")
st.caption("Estimate Medicaid drug reimbursement and explore state-level patterns")

# Tabs 
tab_about, tab_estimator, tab_patterns = st.tabs(["About", "Spending Estimator", "Spending Patterns"])

# ───── About Tab ─────
with tab_about:
    st.subheader("Overview")
    st.markdown("""
    This tool analyzes **Medicaid State Drug Utilization Data** to  
    - Identify high-cost and high-volume drugs  
    - Show spending & usage variation across all states  
    - Estimate reimbursement amounts based on utilization inputs  

    """)

# ───── Spending Estimator Tab ─────
with tab_estimator:
    st.subheader("Estimate Reimbursement Spending")

    col1, col2 = st.columns(2)
    with col1:
        state_full = st.selectbox("State", all_states_full)
    with col2:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])

    # Drug name with autocomplete/search
    product = st.selectbox(
        "Drug Name",
        options=[""] + all_drugs,
        format_func=lambda x: x.title() if x else "Start typing or select a drug...",
        placeholder="Type to search (e.g. ozempic, jardiance, biktarvy)",
        help="Suggestions based on the 683 drugs used in the model. Type to filter.",
        index=0
    )

    col3, col4 = st.columns(2)
    with col3:
        units = st.number_input("Units Reimbursed", min_value=0.0, step=100.0, format="%.0f")
    with col4:
        prescriptions = st.number_input("Number of Prescriptions", min_value=0.0, step=100.0, format="%.0f")

    utilization_intensity = units * prescriptions

    if st.button("Calculate Estimate"):
        if not product.strip():
            st.error("Please select or type a drug name")
        elif units == 0 and prescriptions == 0:
            st.warning("Zero units and prescriptions → estimate may be unreliable")
        else:
            input_df = pd.DataFrame({
                "State Full Name": [state_full],
                "Product Name": [product.lower().strip()],
                "Quarter": [quarter],
                "Units Reimbursed": [units],
                "Number of Prescriptions": [prescriptions],
                "utilization_intensity": [utilization_intensity]
            })

            log_pred = model.predict(input_df)[0]
            spending = np.expm1(log_pred)

            st.success(f"**Estimated spending in {state_full}: ${spending:,.0f}**")

            export_df = input_df.copy()
            export_df["Predicted Spending"] = spending
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download result (CSV)",
                csv,
                "medicaid_estimate.csv",
                "text/csv"
            )

# ───── Spending Patterns Tab ─────
with tab_patterns:
    st.subheader("Spending & Utilization Patterns")

    # Map section
    st.markdown("#### Geographic Overview")
    map_view = st.radio("Show", ["Spending ($ Billion)", "Units Reimbursed (Billion)"], horizontal=True)

    top_n = st.slider("Show only top N states (0 = show all 52)", 0, 52, 0, step=1)

    if map_view == "Spending ($ Billion)":
        df_map = state_spending.copy()
        df_map["Value"] = df_map["Total Amount Reimbursed"] / 1e9
        colorbar_title = "$ Billion"
        colors = "YlOrRd"
    else:
        df_map = state_usage.copy()
        df_map["Value"] = df_map["Units Reimbursed"] / 1e9
        colorbar_title = "Billion units"
        colors = "Blues"

    if top_n > 0:
        df_map = df_map.nlargest(top_n, "Value")

    fig_map = px.choropleth(
        df_map,
        locations="State Code",
        locationmode="USA-states",
        color="Value",
        hover_name="State Full Name",
        hover_data={"Value": ":.2f"},
        color_continuous_scale=colors,
        scope="usa",
        labels={"Value": colorbar_title}
    )

    fig_map.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        height=500,
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Bar chart section
    st.markdown("#### Top Items")
    view = st.selectbox("View", ["Top Spending Drugs", "Most Prescribed Drugs", "Top Spending States", "Top Usage States"])
    n_top = st.slider("Show top", 5, 52, 15)

    if view == "Top Spending Drugs":
        df_plot = top_exp_drugs.head(n_top)
        x, y = "Total Amount Reimbursed", "Product Name"
        title = "Top High-Cost Drugs"
    elif view == "Most Prescribed Drugs":
        df_plot = popular_drugs.head(n_top)
        x, y = "No of prescriptions", "Product Name"
        title = "Most Frequently Prescribed Drugs"
    elif view == "Top Spending States":
        df_plot = state_spending.head(n_top)
        x, y = "Total Amount Reimbursed", "State Full Name"
        title = "Top Spending States"
    else:
        df_plot = state_usage.head(n_top)
        x, y = "Units Reimbursed", "State Full Name"
        title = "Top States by Units Reimbursed"

    fig_bar, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=df_plot, x=x, y=y, ax=ax, palette="Blues_d")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    st.pyplot(fig_bar)


   