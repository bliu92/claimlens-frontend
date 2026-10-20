import re
import requests
import streamlit as st

st.set_page_config(
    page_title="ClaimLens - Two-Stage Refund Decision System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 后端 API 地址配置
INTENT_API_URL = "https://backboard-pulverize-perennial.ngrok-free.dev/predict"
# 第二阶段 API 地址（例如退款理由归类、严重度评估或直接处理结果）
STAGE2_API_URL = "https://backboard-pulverize-perennial.ngrok-free.dev/predict"

# ----------------------------------------------------
# 1. 初始化 Session State 状态管理
# ----------------------------------------------------
if "first_message" not in st.session_state:
    st.session_state.first_message = ""
if "intent_result" not in st.session_state:
    st.session_state.intent_result = None
if "stage2_result" not in st.session_state:
    st.session_state.stage2_result = None


def render_highlighted_text(raw_text: str, keywords: list[str]) -> str:
    """在文本中高亮 SHAP 关键词"""
    if not keywords or not raw_text.strip():
        return raw_text
    sorted_kws = sorted(keywords, key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_kws)) + r')\b', re.IGNORECASE)
    highlight_tag = (
        r'<span style="background-color: #FFE082; color: #B71C1C; '
        r'padding: 2px 6px; margin: 0 2px; border-radius: 4px; font-weight: 600; '
        r'border-bottom: 2px solid #E53935;">\1</span>'
    )
    return pattern.sub(highlight_tag, raw_text)


def reset_workflow():
    """重置工作流状态"""
    st.session_state.first_message = ""
    st.session_state.intent_result = None
    st.session_state.stage2_result = None


# ----------------------------------------------------
# 2. 侧边栏配置
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=64)
    st.title("ClaimLens Console")
    st.markdown("Two-stage NLP pipeline for intent detection & resolution")

    st.markdown("---")
    st.subheader("Model Pipeline")
    st.info("**Stage 1**: TF-IDF + Logistic Regression (Intent)\n\n**Stage 2**: Deep Reasoning / Fine-grained Classifier")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        reset_workflow()
        st.rerun()

    st.markdown("---")
    st.caption("© 2026 ClaimLens Team. All rights reserved.")

# ----------------------------------------------------
# 3. 阶段一：首轮消息输入与意图识别
# ----------------------------------------------------
st.title("🔍 ClaimLens - Customer Service Intelligence")
st.markdown("Stage 1 identifies customer refund intents; if detected, Stage 2 collects further context for decision-making.")

with st.form("stage1_form"):
    st.subheader("Step 1: Customer Initial Message")
    customer_message = st.text_area(
        "Enter customer message",
        value=st.session_state.first_message,
        placeholder="e.g. My package arrived wet and the leather jacket is destroyed. Can I get a refund?",
        height=110
    )

    col1, col2 = st.columns(2)
    with col1:
        submit_predict = st.form_submit_button("⚡ Predict Intent", use_container_width=True)
    with col2:
        submit_shap = st.form_submit_button("🔬 Predict with SHAP", use_container_width=True)

# 处理阶段一预测请求
if submit_predict or submit_shap:
    if not customer_message.strip():
        st.warning("Please enter a customer message first.")
    else:
        st.session_state.first_message = customer_message
        st.session_state.stage2_result = None  # 重置下游结果

        try:
            with st.spinner("Classifying intent..."):
                response = requests.get(
                    INTENT_API_URL,
                    params={"customer_message": customer_message},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

            # 解析并保存到 session_state
            intent = data.get("Refund Intent", "Yes" if data.get("refund_intent") == 1 else "No")
            prob = float(data.get("Refund Intent Probability", data.get("probability", 0.0)))
            model_threshold = float(data.get("threshold", 0.44))
            key_features = data.get("key_features", [])

            st.session_state.intent_result = {
                "intent": intent,
                "probability": prob,
                "threshold": model_threshold,
                "key_features": key_features,
                "mode": "shap" if submit_shap else "standard"
            }
        except requests.exceptions.RequestException as e:
            st.error(f"Stage 1 API Request Failed: {e}")

# ----------------------------------------------------
# 4. 阶段一结果展示
# ----------------------------------------------------
if st.session_state.intent_result:
    res = st.session_state.intent_result
    st.markdown("---")
    st.subheader("📊 Stage 1 Diagnosis")

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        if res["intent"] == "Yes":
            st.error(f"🚨 Intent: **{res['intent']} (Refund Requested)**")
        else:
            st.success(f"💬 Intent: **{res['intent']} (General Inquiry)**")

    with c2:
        st.metric("Refund Probability", f"{res['probability']:.2%}")

    with c3:
        st.metric("Optimal Threshold", f"{res['threshold']:.2f}")

    st.progress(res["probability"])

    # 若选择了 SHAP 模式，展示关键词高亮
    if res["mode"] == "shap" and res["key_features"]:
        st.markdown("##### 🎯 Trigger Keywords (SHAP):")
        highlight_words = [item["word"] for item in res["key_features"]]
        highlighted_html = render_highlighted_text(st.session_state.first_message, highlight_words)
        st.markdown(
            f'<div style="background-color: #F8F9FA; border: 1px solid #E0E0E0; border-radius: 6px; padding: 12px; margin-bottom: 12px;">{highlighted_html}</div>',
            unsafe_allow_html=True
        )

# ----------------------------------------------------
# 5. 阶段二：当 Intent 为 'Yes' 时触发第二轮追问表单
# ----------------------------------------------------
if st.session_state.intent_result and st.session_state.intent_result["intent"] == "Yes":
    st.markdown("---")
    st.markdown("### 📋 Step 2: Additional Case Details (Refund Workflow)")
    st.info("💡 Refund intent confirmed. Please collect or input supplementary details (e.g. order number, item condition, specific reason).")

    with st.form("stage2_form"):
        additional_info = st.text_area(
            "Customer supplementary message / Details",
            placeholder="e.g. Order #883912. The package arrived torn open and the zipper is broken. I already took pictures.",
            height=100
        )

        submit_stage2 = st.form_submit_button("🚀 Submit Complete Case to Model 2", use_container_width=True)

    if submit_stage2:
        if not additional_info.strip():
            st.warning("Please provide additional context for the second model.")
        else:
            # 拼接前后文上下文
            combined_context = f"Initial message: {st.session_state.first_message}\nAdditional details: {additional_info}"

            payload = {
                "initial_message": st.session_state.first_message,
                "additional_message": additional_info,
                "combined_text": combined_context
            }

            try:
                with st.spinner("Processing second-stage reasoning..."):
                    # 也可以用 params GET 请求，结构化数据推荐 POST
                    response_stage2 = requests.post(STAGE2_API_URL, json=payload, timeout=15)
                    response_stage2.raise_for_status()
                    st.session_state.stage2_result = response_stage2.json()

            except requests.exceptions.RequestException as e:
                st.error(f"Stage 2 API Request Failed: {e}")

# ----------------------------------------------------
# 6. 阶段二结果展示
# ----------------------------------------------------
if st.session_state.stage2_result:
    res2 = st.session_state.stage2_result
    st.markdown("---")
    st.subheader("🏁 Stage 2 Final Decision & Routing")

    # 根据你的第二个模型返回结构定制展示，这里给出典型的业务字段示范
    decision = res2.get("decision", res2.get("result", "Processed"))
    category = res2.get("category", "Product Damage")
    suggested_action = res2.get("suggested_action", "Issue instant refund label")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success(f"**Recommended Action**: {suggested_action}")
        st.write(f"**Case Category**: {category}")
    with col_res2:
        st.write(f"**Routing Verdict**: {decision}")

    with st.expander("🔍 View Merged Context Sent to Model 2"):
        st.text(f"Initial: {st.session_state.first_message}")
        st.text(f"Additional: {res2.get('additional_received', 'N/A')}")
