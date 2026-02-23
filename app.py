import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import re
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch
from io import BytesIO

st.title("ICU Structural Digital Twin – Integrated Research Version")

# ---------------------------------------------------
# NORMALIZE MARKDOWN TABLES
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
# TRANSFORMATION (COHERENT MERGE)
# ---------------------------------------------------

def transform(note, intensity):

    clean = normalize_note(note)

    ethical_insert = (
        " Prognosis remains guarded and structured goals-of-care discussion with family is recommended."
    )

    clinical_frame = (
        "Given the above clinical findings, the integrated ICU management plan includes active monitoring and physiologically guided interventions."
    )

    transformed = clinical_frame + " " + clean

    for _ in range(intensity):
        transformed += ethical_insert

    transformed = re.sub(r"\bwas (\w+ed)\b", r"ICU team \1", transformed)

    return transformed

# ---------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------

def monte_carlo(note, runs=30):
    samples = []
    for _ in range(runs):
        perturb = note + np.random.choice(["", " because clinical condition evolved."])
        samples.append(compute_metrics(perturb))
    return np.array(samples)

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

note = st.text_area("Paste ICU Note")
intensity = st.slider("Transformation Intensity", 1, 3, 1)

if st.button("Run Full Structural Simulation"):

    if not note.strip():
        st.warning("Enter note.")
    else:

        original_vec = compute_metrics(note)
        twin_note = transform(note, intensity)
        twin_vec = compute_metrics(twin_note)

        drift = 1 - (np.dot(original_vec, twin_vec) /
                     (norm(original_vec) * norm(twin_vec)))

        ent_orig = entropy(original_vec)
        ent_twin = entropy(twin_vec)

        mc = monte_carlo(note)
        ci = np.percentile(mc, [2.5, 97.5], axis=0)

        data = np.vstack([original_vec, twin_vec])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        kmeans = KMeans(n_clusters=2, random_state=42).fit(data)

        # DISPLAY
        st.subheader("Transformed Note (Merged & Cleaned)")
        st.write(twin_note)

        st.write("Structural Drift:", drift)
        st.write("Original Entropy:", ent_orig)
        st.write("Twin Entropy:", ent_twin)
        st.write("Bootstrap 95% CI:", ci)
        st.write("Cluster Assignment:", kmeans.labels_)

        fig = plt.figure()
        plt.scatter(reduced[:,0], reduced[:,1])
        plt.text(reduced[0,0], reduced[0,1], "Original")
        plt.text(reduced[1,0], reduced[1,1], "Twin")
        st.pyplot(fig)

        # PDF EXPORT
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("ICU Structural Digital Twin Report", styles['Heading1']))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Transformed Note:", styles['Heading3']))
        elements.append(Paragraph(twin_note, styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))

        table_data = [
            ["Metric", "Value"],
            ["Drift", str(drift)],
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
    




