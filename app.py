# app.py
"""Streamlit Dashboard for Automated Fine-Grained Inventory Audit.
Integrated with local Ollama DeepSeek-R1:7b for context reasoning,
severity evaluation, automated audit reports, and interactive staff Q&A.
"""

import streamlit as st
from pathlib import Path
import json
import pandas as pd

# Local modules
from src.object_detection import detect_objects
from src.ocr_engine import read_shelf_tags
from src.planogram_diff import compute_diff
from src.llm_reasoning import generate_report, query_assistant
from src.utils import overlay_boxes

st.set_page_config(page_title="Automated Shelf Audit & Local LLM Reasoning", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("⚙️ Audit & LLM Settings")
fine_grained_mode = st.sidebar.toggle("🔍 Count Minute Items (Pens, Pencils, Clips, Pads)", value=True)

operational_context = st.sidebar.selectbox(
    "📅 Operational Context",
    ["Standard Audit", "Exam Week / Peak Demand", "Back to School Season", "Weekend Sale Clearance"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("🤖 **Local LLM Status**")
st.sidebar.success("DeepSeek-R1:7b (Ollama Local)")

st.title("📦 Automated Shelf Audit & Local DeepSeek-R1 Reasoning")
st.markdown("Combines computer vision structured shelf counts with **local DeepSeek-R1:7b** reasoning for restocking severity, automated checklists, and natural language staff Q&A.")

uploaded_file = st.file_uploader("Upload shelf photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = Path("outputs") / uploaded_file.name
    img_path.write_bytes(uploaded_file.getbuffer())

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("📷 Original Shelf Photo")
        st.image(str(img_path), width="stretch")

    with st.spinner("Processing fine-grained object detection & inventory counting..."):
        detections = detect_objects(str(img_path), fine_grained=fine_grained_mode)
        ocr_results = read_shelf_tags(str(img_path))

        planogram_path = Path("data") / "sample_planogram.json"
        planogram = json.load(planogram_path.open()) if planogram_path.exists() else {}
        diff = compute_diff(detections, ocr_results, planogram)
        annotated_path = overlay_boxes(str(img_path), detections, ocr_results)

    with col_img2:
        st.subheader("🏷️ Numbered Item Count Overlay")
        st.image(str(annotated_path), width="stretch")

    st.markdown("---")

    summary = diff.get("summary", {})
    total_items = summary.get("total_items", 0)
    cat_counts = summary.get("category_counts", {})
    shelf_tags = summary.get("shelf_tags", [])
    items_list = summary.get("items", [])

    st.subheader("📊 Inventory Summary Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📦 Total Physical Items Counted", f"{total_items} units")
    m2.metric("🗂️ Categories Identified", f"{len(cat_counts)} types")
    m3.metric("🏷️ Shelf Section Tags", f"{len(shelf_tags)} tags")
    m4.metric("⚙️ Context Mode", operational_context)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Itemized Stock Count Table",
        "📊 Category Breakdown",
        "🧠 DeepSeek-R1 Audit Report",
        "💬 Ask AI Staff Assistant"
    ])

    with tab1:
        st.subheader("🔢 Itemized Inventory List (Every Single Unit)")
        if items_list:
            df_items = pd.DataFrame([
                {
                    "Item #": item["item_id"],
                    "Label": item["item_label"],
                    "Category": item["category"],
                    "Confidence": item["confidence_str"],
                    "Source": item["source"],
                    "Bounding Box [X, Y, W, H]": str(item["bbox"])
                }
                for item in items_list
            ])
            st.dataframe(df_items, width="stretch", hide_index=True)
        else:
            st.warning("No physical items detected on this shelf.")

        if shelf_tags:
            st.subheader("🏷️ Recognized Shelf Section Tags")
            df_tags = pd.DataFrame([
                {
                    "Tag #": tag["tag_id"],
                    "Text": tag["text"],
                    "Confidence": tag["confidence_str"],
                    "Bounding Box [X, Y, W, H]": str(tag["bbox"])
                }
                for tag in shelf_tags
            ])
            st.dataframe(df_tags, width="stretch", hide_index=True)

    with tab2:
        st.subheader("📊 Item Count Breakdown by Category")
        if cat_counts:
            df_cat = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"]).sort_values(by="Count", ascending=False)
            st.bar_chart(df_cat.set_index("Category"))
            st.table(df_cat)
        else:
            st.info("No category data available.")

    with tab3:
        st.subheader(f"📝 DeepSeek-R1 Audit Report ({operational_context})")
        
        # Check session state for cached report to prevent hanging on page rerun
        report_key = f"report_{uploaded_file.name}_{operational_context}"
        
        col_btn1, col_btn2 = st.columns([2, 5])
        with col_btn1:
            run_llm = st.button("🧠 Refresh DeepSeek-R1 Reasoning", use_container_width=True)

        if run_llm or report_key not in st.session_state:
            with st.spinner(f"Querying local DeepSeek-R1:7b for {operational_context}..."):
                report, thoughts = generate_report(diff, operational_context=operational_context)
                st.session_state[report_key] = (report, thoughts)

        cached_report, cached_thoughts = st.session_state[report_key]
        st.markdown(cached_report)
        if cached_thoughts:
            with st.expander("🔍 Show DeepSeek-R1 Reasoning Chain (<think>)"):
                st.code(cached_thoughts, language="text")

    with tab4:
        st.subheader("💬 Natural-Language AI Staff Query Assistant")
        st.markdown("Ask direct questions to DeepSeek-R1 about current inventory levels, restocking priorities, or shelf discrepancies.")

        col_q1, col_q2, col_q3 = st.columns(3)
        quick_query = None
        if col_q1.button("⚡ What's low on shelf layout?"):
            quick_query = "What items are low or missing on the shelf layout?"
        if col_q2.button("🚀 What is our restock priority today?"):
            quick_query = f"What is our highest restock priority today given context: '{operational_context}'?"
        if col_q3.button("✏️ Are pens or notebooks low for exam week?"):
            quick_query = "Are pens, markers, or notebooks low or in critical supply for exam week?"

        user_query = st.text_input("Enter natural-language question for store staff:", value=quick_query if quick_query else "")

        if user_query:
            with st.spinner("Querying local DeepSeek-R1:7b with live shelf context..."):
                answer, query_thoughts = query_assistant(user_query, diff, operational_context=operational_context)
                st.markdown("### 🤖 DeepSeek-R1 Answer:")
                st.info(answer)
                if query_thoughts:
                    with st.expander("🔍 DeepSeek-R1 Query Reasoning (<think>)"):
                        st.code(query_thoughts, language="text")

else:
    st.info("👈 Upload a shelf photo to run the inventory count and generate an audit report.")

    sample_img = Path("inventory.png")
    if sample_img.exists():
        st.markdown("### Demo Sample Image")
        st.image(str(sample_img), caption="Sample Shelf Image in Workspace", width="stretch")
