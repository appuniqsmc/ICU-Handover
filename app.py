

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import re
from sklearn.decomposition import PCA

# ---------------------------------------------------
# LOAD GLOBAL BASELINE
# ---------------------------------------------------

try:
    global_baseline = np.load("baseline_vector.npy")
except:
    st.error("Missing baseline_vector.npy")
    st.stop()

# Optional institutional baseline
uploaded_file = st.file_uploader("Upload Institutional Baseline (.npy)", type=["npy"])
if uploaded_file:
    institutional_baseline = np.load(uploaded_file)
else:
    institutional_baseline = None

# ---------------------------------------------------
# LEXICONS
# ---------------------------------------------------

ethical_words = [
    "prognosis","goals","family","comfort","palliative",
    "preferences","values","burden","benefit","dnr","dni"
]

decision_verbs = [
    "initiate","start","stop","escalate","withdraw",
    "intubate","dialyze","commence","discontinue"
]

conditional_patterns = [
    r"\bif .*? persists\b",
    r"\bwill review\b",
    r"\bto discuss\b",
    r"\bpending\b",
    r"\breassess\b"
]

causal_connectors = [
    "because","therefore","given","hence",
    "suggests","likely due"
]

# ---------------------------------------------------
# SMART PASSIVE DETECTION
# ---------------------------------------------------

def count_passives(text):
    passive_pattern = r"\b(was|were|is|are|been|being)\b\s+\w+ed\b"
    return len(re.findall(passive_pattern, text.lower()))

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

def count_occurrences(text, word_list):
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in word_list)

def compute_metrics(text):
    word_count = len(text.split())

    ethical_count = count_occurrences(text, ethical_words)
    decision_count = count_occurrences(text, decision_verbs)
    conditional_count = sum(len(re.findall(p, text.lower())) for p in conditional_patterns)
    passive_count = count_passives(text)
    connector_count = count_occurrences(text, causal_connectors)

    DEI = decision_count / (conditional_count + 1)
    AVS = 1 / (passive_count + 1)
    EIR = ethical_count / (word_count + 1)
    ICS = connector_count / (word_count + 1)

    return np.array([DEI, AVS, EIR, ICS])

# ---------------------------------------------------
# MULTI-AXIS TRANSFORMATION ENGINE
# ---------------------------------------------------

def transform(note, axes, intensity):

    text = note

    if "Ethical Integration" in axes:
        insert = (
            " Prognosis remains guarded, and structured goals-of-care "
            "discussion with family is recommended."
        )
        text += insert * int(intensity)

    if "Decision Explicitness" in axes:
        for pattern in conditional_patterns:
            text = re.sub(pattern,
                          "Given clinical persistence, we will initiate active management",
                          text,
                          flags=re.IGNORECASE)

    if "Accountability Visibility" in axes:
        text = re.sub(r"\bwas (\w+ed)\b", r"ICU team \1", text, flags=re.IGNORECASE)

    if "Interpretive Coherence" in axes:
        prefix = "Given the above physiological findings, the integrated management plan is as follows: "
        text = prefix + text

    return text

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.title("Advanced ICU Structural Digital Twin")

note = st.text_area("Paste ICU Handover Note")

axes = st.multiselect(
    "Select Structural Axes",
    [
        "Ethical Integration",
        "Decision Explicitness",
        "Accountability Visibility",
        "Interpretive Coherence"
    ]
)

intensity = st.slider("Transformation Intensity", 0.5, 2.0, 1.0, 0.5)

if st.button("Generate Structural Twin"):

    if not note.strip():
        st.warning("Paste a note.")
    else:

        twin = transform(note, axes, intensity)

        st.subheader("Transformed Note")
        st.write(twin)

        original_vec = compute_metrics(note)
        twin_vec = compute_metrics(twin)

        drift_global = 1 - (np.dot(global_baseline, twin_vec) /
                            (norm(global_baseline) * norm(twin_vec)))

        st.subheader("Drift vs Global Baseline")
        st.write(drift_global)

        if institutional_baseline is not None:
            drift_inst = 1 - (np.dot(institutional_baseline, twin_vec) /
                              (norm(institutional_baseline) * norm(twin_vec)))
            st.subheader("Drift vs Institutional Baseline")
            st.write(drift_inst)

        # PCA Visualization
        data = np.vstack([global_baseline, original_vec, twin_vec])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        fig = plt.figure()
        plt.scatter(reduced[0,0], reduced[0,1])
        plt.scatter(reduced[1,0], reduced[1,1])
        plt.scatter(reduced[2,0], reduced[2,1])

        plt.text(reduced[0,0], reduced[0,1], "Global")
        plt.text(reduced[1,0], reduced[1,1], "Original")
        plt.text(reduced[2,0], reduced[2,1], "Twin")

        st.pyplot(fig)



