import streamlit as st
import pandas as pd
from openai import OpenAI

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(page_title="SAR Generator", layout="wide")
st.title("🕵️ SAR Narrative Generator with Audit Trail")

st.write("Upload a transaction CSV file to generate a Suspicious Activity Report.")

# ===============================
# API KEY INPUT
# ===============================

api_key = st.text_input("Enter OpenAI API Key", type="password")

client = None
if api_key:
    client = OpenAI(api_key=api_key)

# ===============================
# FILE UPLOAD
# ===============================

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Transaction Data Preview")
    st.dataframe(df)

    # ===============================
    # RULE ENGINE
    # ===============================

    triggers = []

    total_amount = df["amount"].sum()
    unique_senders = df["sender_account"].nunique()
    large_transactions = df[df["amount"] > 1000000]

    if total_amount > 4000000:
        triggers.append("High-value aggregated transaction pattern detected.")

    if unique_senders > 30:
        triggers.append("Possible structuring (smurfing) pattern detected.")

    if len(large_transactions) > 0:
        triggers.append("One or more unusually large transactions detected.")

    st.subheader("🚨 Rules Triggered")

    if triggers:
        for t in triggers:
            st.write("✔", t)
    else:
        st.write("No suspicious rules triggered.")

    # ===============================
    # GENERATE SAR
    # ===============================

    if st.button("Generate SAR Narrative"):

        if not client:
            st.error("Please enter OpenAI API key.")
        else:

            prompt = f"""
You are a senior bank compliance officer drafting a Suspicious Activity Report (SAR).

Transaction Summary:
- Total transaction amount: ₹{total_amount}
- Unique sending accounts: {unique_senders}
- Number of transactions: {len(df)}

Rules Triggered:
{triggers}

Write a professional SAR narrative including:
1. Background summary
2. Description of suspicious activity
3. Reason activity is suspicious
4. Recommendation for investigation
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )

                sar_text = response.choices[0].message.content

                st.session_state["sar"] = sar_text
                st.session_state["prompt"] = prompt

            except Exception as e:
                st.error(f"Error generating SAR: {e}")

# ===============================
# DISPLAY & EDIT SAR
# ===============================

if "sar" in st.session_state:

    st.subheader("📝 Generated SAR Narrative")

    edited_sar = st.text_area(
        "Edit SAR Before Approval",
        st.session_state["sar"],
        height=400
    )

    st.session_state["final_sar"] = edited_sar

    if st.button("Approve SAR"):
        st.success("SAR Approved and Ready for Filing.")

# ===============================
# AUDIT TRAIL
# ===============================

if uploaded_file:

    st.subheader("📚 Audit Trail")

    st.write("Total Amount Used:", total_amount)
    st.write("Unique Senders Used:", unique_senders)
    st.write("Number of Transactions:", len(df))
    st.write("Rules Triggered:", triggers)

    if "prompt" in st.session_state:
        st.write("Prompt Sent to LLM:")
        st.code(st.session_state["prompt"])

# ===============================
# DOWNLOAD
# ===============================

if "final_sar" in st.session_state:
    st.download_button(
        "⬇ Download Final SAR",
        st.session_state["final_sar"],
        file_name="SAR_Report.txt"
    )