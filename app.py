import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import re
import pandas as pd
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch
from io import BytesIO

st.title("ICU Structural Digital Twin – Comparative Research Dashboard")

# ---------------------------------------------------
# CLEAN NOTE
# ---------------------------------------------------

def normalize_note(text):
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ---------------------------------------------------
# PASSIVE DETECTION
# ---------------------------------------------------

def count_passives(text):
    pattern = r"\b(was|were|is|are|been|being)\b\s+\w+ed\b"
    return len(re.findall(pattern, text.lower()))

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

def compute_metrics(text):
    word_count = len(text.split()) + 1
    ethical = len(re.findall(r"prognosis|goals|family|palliative", text.lower()))
    decision = len(re.findall(r"initiate|start|stop|intubate|withdraw", text.lower()))
    conditional = len(re.findall(r"if .*? persists|will review|pending|reassess", text.lower()))
    passive = count_passives(text)
    connector = len(re.findall(r"because|therefore|given|hence", text.lower()))

    DEI = decision / (conditional + 1)
    AVS = 1 / (passive + 1)
    EIR = ethical / word_count
    ICS = connector / word_count

    return np.array([DEI, AVS, EIR, ICS])

# ---------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------

def ethical_transform(text):
    return text + " Prognosis remains guarded and structured goals-of-care discussion with family is recommended."

def decision_transform(text):
    text = re.sub(r"\bwill review\b", "We will actively review and intervene", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpending\b", "Actively awaiting with intervention plan", text, flags=re.IGNORECASE)
    return text

def accountability_transform(text):
    return re.sub(r"\bwas (\w+ed)\b", r"ICU team \1", text)

def coherence_transform(text):
    prefix = "Given the above physiological findings, the integrated ICU management plan is as follows: "
    return prefix + text

def merged_transform(text):
    text = ethical_transform(text)
    text = decision_transform(text)
    text = accountability_transform(text)
    text = coherence_transform(text)
    return text

# ---------------------------------------------------
# CORPUS BASELINE
# ---------------------------------------------------

uploaded_corpus = st.file_uploader("Upload corpus CSV (column: note)", type=["csv"])

if uploaded_corpus:
    df = pd.read_csv(uploaded_corpus)
    corpus_vectors = np.array([compute_metrics(n) for n in df["note"]])
    corpus_baseline = np.mean(corpus_vectors, axis=0)
    st.success("Corpus baseline generated.")
else:
    corpus_baseline = None

# ---------------------------------------------------
# MAIN INPUT
# ---------------------------------------------------

note = st.text_area("Paste ICU Note")

if st.button("Run Comparative Structural Analysis"):

    if not note.strip():
        st.warning("Enter note.")
    else:

        clean = normalize_note(note)

        twins = {
            "Original": clean,
            "Ethical": ethical_transform(clean),
            "Decision": decision_transform(clean),
            "Accountability": accountability_transform(clean),
            "Coherence": coherence_transform(clean),
            "Merged": merged_transform(clean)
        }

        metrics = {}
        for key in twins:
            metrics[key] = compute_metrics(twins[key])

        # ---------------------------------------------------
        # MULTI-PANEL DISPLAY
        # ---------------------------------------------------

        for key in twins:
            with st.expander(key + " Version"):
                st.write(twins[key])
                st.write("Metrics:", metrics[key])

        # ---------------------------------------------------
        # RADAR OVERLAY
        # ---------------------------------------------------

        st.subheader("Radar Structural Comparison")

        labels = ["DEI","AVS","EIR","ICS"]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        for key in metrics:
            values = list(metrics[key]) + [metrics[key][0]]
            ax.plot(angles, values, label=key)

        if corpus_baseline is not None:
            baseline_vals = list(corpus_baseline) + [corpus_baseline[0]]
            ax.plot(angles, baseline_vals, linestyle='dashed', label="Corpus Baseline")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))

        st.pyplot(fig)

        # ---------------------------------------------------
        # PCA VISUALIZATION
        # ---------------------------------------------------

        st.subheader("PCA Structural Embedding")

        data = np.vstack([metrics[k] for k in metrics])

        if corpus_baseline is not None:
            data = np.vstack([data, corpus_baseline])

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        fig2 = plt.figure()
        for i, key in enumerate(metrics):
            plt.scatter(reduced[i,0], reduced[i,1])
            plt.text(reduced[i,0], reduced[i,1], key)

        if corpus_baseline is not None:
            plt.scatter(reduced[-1,0], reduced[-1,1])
            plt.text(reduced[-1,0], reduced[-1,1], "Baseline")

        st.pyplot(fig2)

        # ---------------------------------------------------
        # PDF EXPORT (FOR MERGED MODE)
        # ---------------------------------------------------

        selected_for_pdf = st.selectbox("Select Mode to Export PDF", list(twins.keys()))

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("ICU Structural Digital Twin Report", styles['Heading1']))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Mode: {selected_for_pdf}", styles['Heading3']))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(twins[selected_for_pdf], styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))

        table_data = [
            ["Metric", "Value"],
            ["DEI", str(metrics[selected_for_pdf][0])],
            ["AVS", str(metrics[selected_for_pdf][1])],
            ["EIR", str(metrics[selected_for_pdf][2])],
            ["ICS", str(metrics[selected_for_pdf][3])]
        ]

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))

        elements.append(table)
        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            "Download Selected Mode PDF",
            data=buffer,
            file_name="ICU_structural_report.pdf",
            mime="application/pdf"
        )






