import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

st.set_page_config(
    page_title="AI Compliance Intelligence Platform",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #000000;
}
h1 { 
    color: #000000; 
    font-weight: 700;
    border-bottom: 3px solid #000000;
    padding-bottom: 10px;
}
h2, h3 {
    color: #1a1a1a;
}
[data-testid="stSidebar"] { 
    background-color: #f8f9fa;
    border-right: 2px solid #e0e0e0;
}
[data-testid="metric-container"] {
    background: #ffffff;
    border: 2px solid #000000;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #000000;
    color: white;
    border-radius: 6px;
    font-weight: 600;
    border: 2px solid #000000;
}
.stButton>button:hover {
    background-color: #333333;
    border-color: #333333;
}
[data-testid="stExpander"] {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
}
.stTextArea textarea {
    border: 2px solid #000000;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #f8f9fa;
    border: 1px solid #000000;
    color: #000000;
}
.stTabs [aria-selected="true"] {
    background-color: #000000;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡 AI Compliance Intelligence Platform")
st.caption("Privacy-First | Offline LLM | AML Risk Engine")

with st.expander("ℹ️ How to use this platform", expanded=False):
    st.markdown("""
    ### Generation Modes:
    
    **⚡ Fast Template (Recommended)**
    - Instant professional SAR reports
    - No AI required - works offline
    - Fully compliant with regulatory standards
    - Perfect for quick reports
    
    **🤖 AI Enhanced**
    - Requires Ollama running locally
    - Adds AI-generated narrative analysis
    - Max 30-second timeout with template fallback
    - Install Ollama: `https://ollama.ai`
    - Run model: `ollama run phi3:mini`
    
    ### Quick Start:
    1. Enter officer name and branch (optional)
    2. Upload your transaction CSV
    3. Review risk metrics
    4. Click "Generate SAR"
    5. Edit and download
    """)

with st.sidebar:
    st.header("⚙ Configuration")
    
    st.subheader("👤 Report Details")
    officer_name = st.text_input("Compliance Officer Name", value="", placeholder="e.g., John Smith")
    bank_branch = st.text_input("Bank Branch/Location", value="", placeholder="e.g., New York Main Branch")
    
    st.markdown("---")
    st.subheader("🤖 AI Settings")

    model_name = st.selectbox("LLM Model", ["phi3:mini", "mistral", "llama3"])
    regime = st.selectbox("Regulatory Regime", ["FinCEN (US)", "FATF", "RBI (India)"])

    st.markdown("---")
    st.subheader("🎯 Risk Thresholds")
    
    high_volume_threshold = st.number_input("High Total Volume Threshold", value=4000000)
    large_tx_threshold = st.number_input("Large Transaction Threshold", value=1000000)
    sender_threshold = st.number_input("Unique Sender Threshold", value=30)

    risk_weight_volume = st.slider("Risk Weight: Volume", 1, 5, 3)
    risk_weight_sender = st.slider("Risk Weight: Senders", 1, 5, 3)
    risk_weight_large = st.slider("Risk Weight: Large TX", 1, 5, 4)

    st.markdown("---")
    generation_mode = st.radio(
        "SAR Generation Mode",
        ["⚡ Fast Template (Instant)", "🤖 AI Enhanced (30s)"],
        help="Template mode generates instantly, AI mode uses LLM for enrichment"
    )
    
    advanced_mode = st.checkbox("Enable Detailed AI Analysis") if generation_mode == "🤖 AI Enhanced (30s)" else False

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

def generate_sar_ai(prompt, model):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model, 
        "prompt": prompt, 
        "stream": False,
        "options": {
            "num_predict": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 1024
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=25)
        if response.status_code == 200:
            result = response.json().get("response", "")
            if result and len(result.strip()) > 20:
                return result.strip()
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return None

def generate_sar_template(total_amount, unique_senders, tx_count, triggers, regime, risk_score):
    trigger_text = "\n".join([f"• {t}" for t in triggers])
    
    if risk_score >= 8:
        assessment = "CRITICAL - Immediate regulatory filing recommended. Pattern indicates sophisticated financial crime activity requiring urgent investigation."
        action = "File SAR within 24 hours. Freeze suspicious accounts pending investigation. Escalate to senior compliance officer."
    elif risk_score >= 4:
        assessment = "ELEVATED - Enhanced due diligence required. Activity exhibits characteristics consistent with money laundering typologies."
        action = "Conduct enhanced customer due diligence. Monitor account activity closely. Prepare SAR filing documentation."
    else:
        assessment = "MODERATE - Warrants monitoring and documentation. Some unusual patterns detected but below critical threshold."
        action = "Continue enhanced monitoring. Document findings in compliance log. Review in next audit cycle."
    
    narrative = f"""The financial institution has identified suspicious transaction patterns warranting regulatory review. 
Analysis reveals {tx_count} transactions totaling ${total_amount:,.2f} across {unique_senders} unique sender accounts. 

The activity triggered multiple AML detection rules including: {', '.join([t.lower() for t in triggers])}. 

{assessment}

Pattern analysis indicates potential structuring behavior designed to evade reporting thresholds. The velocity and distribution of transactions are inconsistent with legitimate business activity for this account profile.

Regulatory framework: {regime} compliance standards applied."""

    return narrative, action

def generate_pdf(report_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, topMargin=0.75*inch, bottomMargin=0.75*inch)
    elements = []
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "Header",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#000000"),
        spaceAfter=16,
        alignment=1,
        fontName="Helvetica-Bold"
    )
    
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=8,
        spaceBefore=12,
        fontName="Helvetica-Bold"
    )
    
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6,
        leading=14,
        fontName="Helvetica"
    )
    
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        leftIndent=20,
        spaceAfter=4,
        leading=14,
        fontName="Helvetica"
    )

    elements.append(Paragraph("SUSPICIOUS ACTIVITY REPORT", header_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#000000")))
    elements.append(Spacer(1, 0.3 * inch))

    lines = report_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            elements.append(Spacer(1, 0.1 * inch))
            continue
        
        if line.endswith(':') and line.isupper() and len(line) < 50:
            elements.append(Paragraph(line, section_header_style))
            current_section = line
        
        elif line.startswith('•') or line.startswith('-'):
            clean_line = line.lstrip('•- ').strip()
            elements.append(Paragraph(f"• {clean_line}", bullet_style))
        
        elif ':' in line and len(line.split(':')[0]) < 40:
            key, value = line.split(':', 1)
            formatted = f"<b>{key}:</b> {value.strip()}"
            elements.append(Paragraph(formatted, body_style))
        
        elif line == "SUSPICIOUS ACTIVITY REPORT":
            continue
        
        else:
            elements.append(Paragraph(line, body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def get_risk_level(score):
    if score >= 8:
        return "🔴 HIGH RISK"
    elif score >= 4:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = {"amount", "sender_account"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain: amount, sender_account")
        st.stop()

    total_amount = df["amount"].sum()
    unique_senders = df["sender_account"].nunique()
    large_tx = df[df["amount"] > large_tx_threshold]

    risk_score = 0
    triggers = []

    if total_amount > high_volume_threshold:
        triggers.append("High aggregated transaction volume")
        risk_score += risk_weight_volume

    if unique_senders > sender_threshold:
        triggers.append("Possible structuring (smurfing) pattern")
        risk_score += risk_weight_sender

    if len(large_tx) > 0:
        triggers.append("Unusually large transaction detected")
        risk_score += risk_weight_large

    risk_level = get_risk_level(risk_score)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Amount", f"${total_amount:,.0f}")
    col2.metric("Transactions", len(df))
    col3.metric("Unique Senders", unique_senders)
    col4.metric("Risk Score", risk_score)
    col5.metric("Risk Level", risk_level)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        gauge={
            'axis': {'range': [0, 15]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 4], 'color': "#e0e0e0"},
                {'range': [4, 8], 'color': "#bdbdbd"},
                {'range': [8, 15], 'color': "#9e9e9e"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='white',
        font={'color': "black", 'family': "Arial"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Full Transaction Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("⚠ Triggered Rules")
    if triggers:
        for t in triggers:
            st.success(t)
    else:
        st.info("No suspicious activity detected.")

    if triggers:
        if st.button("🚀 Generate Professional SAR", type="primary", use_container_width=True):
            
            use_ai = generation_mode == "🤖 AI Enhanced (30s)"
            
            if use_ai:
                try:
                    test_response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    ollama_available = test_response.status_code == 200
                except:
                    ollama_available = False
                
                if not ollama_available:
                    st.warning("⚠️ Ollama LLM not detected. Using fast template mode.")
                    use_ai = False
            
            if use_ai:
                short_prompt = f"""Write 2-3 sentences analyzing suspicious financial activity:
Amount: ${total_amount:,.0f}, Senders: {unique_senders}, Issues: {', '.join(triggers[:2])}
Focus on money laundering risk."""

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🤖 Connecting to AI model...")
                progress_bar.progress(20)
                
                ai_narrative = generate_sar_ai(short_prompt, model_name)
                
                progress_bar.progress(100)
                
                if ai_narrative and len(ai_narrative) > 20:
                    narrative = ai_narrative
                    action = f"Enhanced Due Diligence and Regulatory Filing as per {regime}."
                    status_text.empty()
                    progress_bar.empty()
                    st.success("✅ AI-enhanced narrative generated successfully!")
                else:
                    status_text.empty()
                    progress_bar.empty()
                    st.warning("⏱️ AI took too long - generated professional template report instead")
                    st.info("💡 Tip: Try 'Fast Template' mode for instant generation, or check if Ollama is running")
                    narrative, action = generate_sar_template(
                        total_amount, unique_senders, len(df), triggers, regime, risk_score
                    )
            else:
                with st.spinner("⚡ Generating report..."):
                    narrative, action = generate_sar_template(
                        total_amount, unique_senders, len(df), triggers, regime, risk_score
                    )
                st.success("✅ Professional SAR generated instantly!")

            professional_format = f"""SUSPICIOUS ACTIVITY REPORT
Reporting Institution: {bank_branch if bank_branch else "[Bank Branch/Location]"}
Compliance Officer: {officer_name if officer_name else "[Officer Name]"}
Report Date: {datetime.utcnow().strftime('%Y-%m-%d')}

EXECUTIVE SUMMARY:
Risk Level: {risk_level}
Risk Score: {risk_score}/15

TRANSACTION OVERVIEW:
Total Transaction Volume: ${total_amount:,.2f}
Total Transactions Reviewed: {len(df)}
Unique Senders: {unique_senders}
Large Transactions (>${large_tx_threshold:,}): {len(large_tx)}

TRIGGERED RULES:
{chr(10).join(['• ' + t for t in triggers])}

NARRATIVE ANALYSIS:
{narrative}

RECOMMENDED ACTION:
{action}

Regulatory Framework: {regime}
Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

            st.session_state["sar"] = professional_format

            st.session_state["audit"] = {
                "timestamp": str(datetime.utcnow()),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "triggers": triggers,
                "generation_mode": generation_mode,
                "ai_used": use_ai and ai_narrative is not None
            }

if "sar" in st.session_state:
    st.markdown("---")
    st.subheader("📝 Professional SAR - Review & Edit")
    
    tab1, tab2 = st.tabs(["✏️ Edit Report", "👁️ Preview"])
    
    with tab1:
        st.info("💡 **Tip**: Edit the report directly below. Your changes will be reflected in downloads.")
        
        edited = st.text_area(
            "Report Content", 
            st.session_state["sar"], 
            height=450,
            help="Edit any section of the report. Changes auto-save for download."
        )
        
        st.session_state["sar_edited"] = edited
        
        st.markdown("#### 🔧 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("📋 Copy to Clipboard"):
            st.code(edited, language=None)
            st.success("✅ Report ready to copy!")
        
        if col2.button("🔄 Reset to Original"):
            st.session_state["sar_edited"] = st.session_state["sar"]
            st.rerun()
        
        word_count = len(edited.split())
        col3.metric("Word Count", word_count)
        
        char_count = len(edited)
        col4.metric("Characters", char_count)
    
    with tab2:
        st.markdown("### 📄 Preview (Read-Only)")
        st.text(st.session_state.get("sar_edited", edited))

    st.markdown("---")
    st.subheader("⬇️ Download Options")
    
    final_report = st.session_state.get("sar_edited", edited)
    
    col1, col2, col3 = st.columns(3)

    col1.download_button(
        "📄 Download TXT",
        final_report,
        file_name=f"SAR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        use_container_width=True
    )

    col2.download_button(
        "📋 Download JSON",
        json.dumps({
            "SAR": final_report,
            "metadata": st.session_state.get("audit", {})
        }, indent=4),
        file_name=f"SAR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        use_container_width=True
    )

    pdf_file = generate_pdf(final_report)

    col3.download_button(
        "📕 Download PDF",
        pdf_file,
        file_name=f"SAR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.markdown("---")
    if st.button("🗑️ Clear Session & Start New Report", type="secondary"):
        st.session_state.clear()
        st.rerun()
