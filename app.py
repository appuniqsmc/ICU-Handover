import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from openai import OpenAI

# -------------------------
# LOAD BASELINE VECTOR
# -------------------------

baseline_vector = np.load("baseline_vector.npy")

# -------------------------
# LEXICONS
# -------------------------

ethical_lexicon = [
    "prognosis","goals","family","comfort","palliative",
    "preferences","values","burden","benefit","dnr","dni"
]

decision_verbs = [
    "initiate","start","stop","escalate","withdraw",
    "intubate","dialyze","commence","discontinue"
]

conditional_phrases = [
    "if persists","to discuss","will review",
    "await","pending","reassess","consider","if worsening"
]

explicit_agents = [
    "we","icu team","discussed","decided"
]

causal_connectors = [
    "because","therefore","given","hence",
    "suggests","likely due"
]

# -------------------------
# METRIC FUNCTIONS
# -------------------------

def count_occurrences(text, word_list):
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in word_list)

def count_passives(text):
    return text.lower().count("was ")

def compute_metrics(text):
    word_count = len(text.split())

    ethical_count = count_occurrences(text, ethical_lexicon)
    decision_count = count_occurrences(text, decision_verbs)
    conditional_count = count_occurrences(text, conditional_phrases)
    agent_count = count_occurrences(text, explicit_agents)
    passive_count = count_passives(text)
    connector_count = count_occurrences(text, causal_connectors)

    DEI = decision_count / (conditional_count + 1)
    AVS = agent_count / (passive_count + 1)
    EIR = ethical_count / (word_count + 1)
    ICS = connector_count / (word_count + 1)

    return {
        "DEI": DEI,
        "AVS": AVS,
        "EIR": EIR,
        "ICS": ICS
    }

# -------------------------
# PROMPTS
# -------------------------

def get_prompt(note, style):

    if style == "Ethical Integration":
        return f"""
        Rewrite this ICU handover note.
        Preserve clinical facts.
        Increase ethical integration and goals-of-care framing.
        Do not add new clinical data.

        NOTE:
        {note}
        """

    if style == "Decision Explicitness":
        return f"""
        Rewrite this ICU handover note.
        Preserve clinical facts.
        Convert deferred language into explicit decisions.
        Do not add new data.

        NOTE:
        {note}
        """

    if style == "Accountability Visibility":
        return f"""
        Rewrite this ICU handover note.
        Preserve clinical facts.
        Increase explicit identification of responsible agents.
        Reduce passive constructions.
        Do not invent actions.

        NOTE:
        {note}
        """

    if style == "Interpretive Coherence":
        return f"""
        Rewrite this ICU handover note.
        Preserve clinical facts.
        Integrate reasoning connectors.
        Avoid fragmented listing.
        Do not add new data.

        NOTE:
        {note}
        """

# -------------------------
# OPENAI CLIENT
# -------------------------

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_twin(note, style):
    prompt = get_prompt(note, style)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content

# -------------------------
# STREAMLIT UI
# -------------------------

st.title("ICU Documentation Digital Twin")

note = st.text_area("Paste ICU Handover Note")

style = st.selectbox(
    "Select Structural Axis",
    [
        "Ethical Integration",
        "Decision Explicitness",
        "Accountability Visibility",
        "Interpretive Coherence"
    ]
)

if st.button("Generate Twin"):

    if note.strip() == "":
        st.warning("Please paste a note.")
    else:

        transformed_note = generate_twin(note, style)

        st.subheader("Transformed Note")
        st.write(transformed_note)

        original_metrics = compute_metrics(note)
        twin_metrics = compute_metrics(transformed_note)

        original_vector = np.array([
            original_metrics["DEI"],
            original_metrics["AVS"],
            original_metrics["EIR"],
            original_metrics["ICS"]
        ])

        twin_vector = np.array([
            twin_metrics["DEI"],
            twin_metrics["AVS"],
            twin_metrics["EIR"],
            twin_metrics["ICS"]
        ])

        drift = 1 - (np.dot(baseline_vector, twin_vector) /
                     (norm(baseline_vector) * norm(twin_vector)))

        st.subheader("Metrics")
        st.write("Original:", original_metrics)
        st.write("Twin:", twin_metrics)
        st.write("Structural Drift:", drift)

        labels = ["DEI","AVS","EIR","ICS"]

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        original_vals = list(original_vector) + [original_vector[0]]
        twin_vals = list(twin_vector) + [twin_vector[0]]

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        ax.plot(angles, original_vals, label="Original")
        ax.plot(angles, twin_vals, label="Twin")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        ax.legend()
        st.pyplot(fig)
