import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch
from io import BytesIO

st.title("ICU Structural Digital Twin – Modular Research Engine")

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
# ENTROPY
# ---------------------------------------------------

def entropy(vec):
    vec = vec + 1e-8
    p = vec / np.sum(vec)
    return -np.sum(p * np.log(p))

# ---------------------------------------------------
# INDIVIDUAL TRANSFORMATIONS
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

# ---------------------------------------------------
# MERGED TRANSFORMATION
# ---------------------------------------------------

def merged_transform(text):
    text = ethical_transform(text)
    text = decision_transform(text)
    text = accountability_transform(text)
    text = coherence_transform(text)
    return text

# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------

note = st.text_area("Paste ICU Note")

mode = st.selectbox(
    "Select Structural Mode",
    [
        "Ethical Integration",
        "Decision Explicitness",
        "Accountability Visibility",
        "Interpretive Coherence",
        "Merged (All Axes)"
    ]
)

if st.button("Run Structural Simulation"):

    if not note.strip():
        st.warning("Enter note.")
    else:

        clean_note = normalize_note(note)

        if mode == "Ethical Integration":
            twin = ethical_transform(clean_note)

        elif mode == "Decision Explicitness":
            twin = decision_transform(clean_note)

        elif mode == "Accountability Visibility":
            twin = accountability_transform(clean_note)

        elif mode == "Interpretive Coherence":
            twin = coherence_transform(clean_note)

        else:
            twin = merged_transform(clean_note)

        # Metrics
        original_vec = compute_metrics(clean_note)
        twin_vec = compute_metrics(twin)

        drift = 1 - (np.dot(original_vec, twin_vec) /
                     (norm(original_vec) * norm(twin_vec)))

        ent_orig = entropy(original_vec)
        ent_twin = entropy(twin_vec)

        # PCA
        data = np.vstack([original_vec, twin_vec])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42).fit(data)

        # Display
        st.subheader("Transformed Note")
        st.write(twin)

        st.write("Structural Drift:", drift)
        st.write("Original Entropy:", ent_orig)
        st.write("Twin Entropy:", ent_twin)
        st.write("Cluster Assignment:", kmeans.labels_)

        fig = plt.figure()
        plt.scatter(reduced[:,0], reduced[:,1])
        plt.text(reduced[0,0], reduced[0,1], "Original")
        plt.text(reduced[1,0], reduced[1,1], "Twin")
        st.pyplot(fig)

        # ---------------------------------------------------
        # PDF EXPORT
        # ---------------------------------------------------

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("ICU Structural Digital Twin Report", styles['Heading1']))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Mode: {mode}", styles['Heading3']))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Transformed Note:", styles['Heading3']))
        elements.append(Paragraph(twin, styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))

        table_data = [
            ["Metric", "Value"],
            ["Structural Drift", str(drift)],
            ["Original Entropy", str(ent_orig)],
            ["Twin Entropy", str(ent_twin)]
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
            "Download PDF Report",
            data=buffer,
            file_name="ICU_structural_report.pdf",
            mime="application/pdf"
        )
        )
    





