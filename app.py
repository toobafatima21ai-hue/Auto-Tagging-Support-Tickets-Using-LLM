# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from config import CATEGORIES

st.set_page_config(page_title="Support Ticket Auto-Tagger", layout="wide")

# ---------- Cached model loaders ----------
@st.cache_resource
def load_zero_shot():
    from zero_shot import tag_ticket_zero_shot
    return tag_ticket_zero_shot

@st.cache_resource
def load_finetuned():
    from inference_finetuned import tag_ticket_finetuned
    return tag_ticket_finetuned

def get_few_shot():
    from few_shot import tag_ticket_few_shot
    return tag_ticket_few_shot


st.title("🎫 Support Ticket Auto-Tagger")
st.caption("Zero-shot vs Few-shot vs Fine-tuned LLM classification — top 3 tags per ticket")

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2, tab3 = st.tabs(["🔎 Single Ticket", "📂 Batch Tagging", "📊 Analytics"])

METHODS = ["Fine-Tuned (DistilBERT)", "Zero-Shot (BART-MNLI)", "Few-Shot (Ollama phi3)", "Compare All"]

# ---------------- TAB 1: Single ticket ----------------
with tab1:
    method = st.selectbox("Tagging method", METHODS)
    text = st.text_area("Paste a support ticket:", height=120,
                         placeholder="e.g. My subscription was charged twice this month...")

    if st.button("Tag Ticket", type="primary") and text.strip():
        results = {}
        with st.spinner("Tagging..."):
            if method in ("Fine-Tuned (DistilBERT)", "Compare All"):
                results["Fine-Tuned"] = load_finetuned()(text, top_k=3)
            if method in ("Zero-Shot (BART-MNLI)", "Compare All"):
                results["Zero-Shot"] = load_zero_shot()(text, top_k=3)
            if method in ("Few-Shot (Ollama phi3)", "Compare All"):
                try:
                    results["Few-Shot"] = get_few_shot()(text, top_k=3)
                except Exception as e:
                    st.warning(f"Ollama not reachable — start it with `ollama serve`. ({e})")

        cols = st.columns(len(results)) if results else []
        for col, (name, preds) in zip(cols, results.items()):
            with col:
                st.subheader(name)
                df = pd.DataFrame(preds, columns=["Tag", "Confidence"])
                st.dataframe(df, hide_index=True, use_container_width=True)
                fig = px.bar(df, x="Confidence", y="Tag", orientation="h",
                             range_x=[0, 1], color="Confidence",
                             color_continuous_scale="Blues")
                fig.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

        st.session_state.history.append({
            "text": text, "method": method,
            "top_tag": next(iter(results.values()))[0][0] if results else None,
        })

# ---------------- TAB 2: Batch tagging ----------------
with tab2:
    st.write("Upload a CSV with a text column of tickets to tag in bulk.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_method = st.selectbox("Method for batch tagging", METHODS[:3], key="batch_method")

    if uploaded:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Which column contains the ticket text?", df.columns)

        if st.button("Tag All Tickets", type="primary"):
            fn = {
                "Fine-Tuned (DistilBERT)": load_finetuned(),
                "Zero-Shot (BART-MNLI)": load_zero_shot(),
                "Few-Shot (Ollama phi3)": get_few_shot(),
            }[batch_method]

            progress = st.progress(0, text="Tagging tickets...")
            tag1, tag2, tag3, conf1 = [], [], [], []
            for i, row in enumerate(df[text_col].astype(str)):
                preds = fn(row, top_k=3)
                labels = [p[0] for p in preds] + [""] * (3 - len(preds))
                tag1.append(labels[0]); tag2.append(labels[1]); tag3.append(labels[2])
                conf1.append(round(preds[0][1], 3) if preds else 0)
                progress.progress((i + 1) / len(df), text=f"Tagging tickets... {i+1}/{len(df)}")

            df["top_tag_1"], df["top_tag_2"], df["top_tag_3"] = tag1, tag2, tag3
            df["confidence"] = conf1
            st.session_state.batch_result = df
            progress.empty()
            st.success(f"Tagged {len(df)} tickets.")

    if "batch_result" in st.session_state:
        st.dataframe(st.session_state.batch_result, use_container_width=True)
        csv = st.session_state.batch_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Tagged CSV", csv, "tagged_tickets.csv", "text/csv")

# ---------------- TAB 3: Analytics ----------------
with tab3:
    if "batch_result" in st.session_state:
        df = st.session_state.batch_result
        c1, c2 = st.columns(2)
        with c1:
            dist = df["top_tag_1"].value_counts().reset_index()
            dist.columns = ["Tag", "Count"]
            fig = px.pie(dist, names="Tag", values="Count", title="Primary Tag Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x="confidence", nbins=20, title="Confidence Score Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        st.metric("Average confidence", f"{df['confidence'].mean():.2f}")
        st.metric("Tickets tagged", len(df))
    else:
        st.info("Run batch tagging first to see analytics here.")

with st.sidebar:
    st.header("About")
    st.write("Compares 3 LLM-based approaches for support ticket tagging:")
    st.markdown("- **Zero-shot**: BART-MNLI, no training\n"
                "- **Few-shot**: Local LLM (Ollama phi3:mini) with prompt examples\n"
                "- **Fine-tuned**: DistilBERT trained on labeled tickets")
    st.write("Categories:")
    st.write(", ".join(CATEGORIES))
