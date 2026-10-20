import re
import requests
import streamlit as st

st.set_page_config(
    page_title="ClaimLens - Intelligent recognition of customer refund intent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

url = "https://backboard-pulverize-perennial.ngrok-free.dev/predict"


def render_highlighted_text(raw_text: str, keywords: list[str]) -> str:
    """在文本中对提取出的 SHAP 关键词进行视觉高亮"""
    if not keywords or not raw_text.strip():
        return raw_text

    # 优先匹配较长的短语，避免子词干扰
    sorted_kws = sorted(keywords, key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_kws)) + r')\b', re.IGNORECASE)
    highlight_tag = (
        r'<span style="background-color: #FFE082; color: #B71C1C; '
        r'padding: 2px 6px; margin: 0 2px; border-radius: 4px; font-weight: 600; '
        r'border-bottom: 2px solid #E53935;">\1</span>'
    )
    return pattern.sub(highlight_tag, raw_text)


# ==========================================
# 侧边栏配置
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=64)
    st.title("ClaimLens Console")
    st.markdown("Refund intent recognition based on NLP pipeline processing")

    st.markdown("---")
    st.subheader("Model Selection")
    st.info("Current Model: **TF-IDF (1-2 gram) + Logistic Regression**")

    # 动态阈值滑块
    threshold = st.slider(
        "Intent Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.44,
        step=0.01,
        help="Probability threshold to classify a message as a refund intent"
    )

    st.markdown("---")
    st.caption("© 2026 ClaimLens Team. All rights reserved.")

# ==========================================
# 主界面表单
# ==========================================
st.title("🔍 ClaimLens - Customer Refund Intent Detector")
st.markdown("Enter a customer support message below to detect whether they are requesting a return/refund.")

with st.form("Customer message"):
    customer_message = st.text_area(
        "Customer message",
        value="I received the jacket yesterday but the size is too small, can I return and get a refund?",
        height=120
    )

    # 并排两个提交按钮
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        submit_standard = st.form_submit_button("⚡ Predict", use_container_width=True)
    with col_btn2:
        submit_shap = st.form_submit_button("🔬 Predict with SHAP", use_container_width=True)

# ==========================================
# 预测与结果渲染
# ==========================================
if submit_standard or submit_shap:
    if not customer_message.strip():
        st.warning("Please enter a customer message first.")
    else:
        params = {
            "customer_message": customer_message,
            "threshold": threshold
        }

        try:
            with st.spinner("Analyzing message..."):
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                result = response.json()

            # 兼容两种返回格式
            intent = result.get("Refund Intent", "Yes" if result.get("refund_intent") == 1 else "No")
            prob = result.get("Refund Intent Probability", result.get("probability", 0.0))
            key_features = result.get("key_features", [])

            st.markdown("---")
            st.subheader("Results")

            # 状态展示
            col_metric1, col_metric2 = st.columns([1, 1])
            with col_metric1:
                if intent == "Yes":
                    st.error(f"🚨 Predicted Refund Intent: **{intent}**")
                else:
                    st.success(f"💬 Predicted Refund Intent: **{intent}**")

            with col_metric2:
                st.metric(label="Refund Probability", value=f"{prob:.2%}")
                st.progress(float(prob))

            # 仅在点击 "Predict with SHAP" 时渲染关键词及高亮
            if submit_shap:
                st.markdown("### 🎯 Key Explainability Features (SHAP)")

                if key_features:
                    highlight_words = [item["word"] for item in key_features]

                    # 1. 文本高亮展示
                    highlighted_html = render_highlighted_text(customer_message, highlight_words)
                    st.markdown(
                        f"""
                        <div style="background-color: #F8F9FA; border: 1px solid #E0E0E0;
                                    border-radius: 8px; padding: 16px; font-size: 16px; line-height: 1.8;">
                            {highlighted_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # 2. 核心贡献特征词云标签
                    st.markdown("##### 📌 Top Contributing Keywords & Scores:")
                    cols = st.columns(min(len(key_features), 6))
                    for i, item in enumerate(key_features[:6]):
                        with cols[i]:
                            st.badge(f"**{item['word']}**", icon="🔥")
                            st.caption(f"+{item['score']:.4f}")
                else:
                    st.info("No significant keywords found pushing the prediction toward a refund intent.")

        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
