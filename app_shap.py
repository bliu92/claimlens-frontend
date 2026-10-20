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

    # 静态提示：告知用户阈值由模型自适应提供
    st.markdown("##### Decision Strategy")
    st.caption("⚡ **Adaptive Threshold**: Determined automatically by model validation optimization.")

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
        # 不传 threshold 参数，完全由后端模型提供默认决策阈值
        params = {
            "customer_message": customer_message
        }

        try:
            with st.spinner("Analyzing message..."):
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                result = response.json()

            # 解析后端 API 返回数据
            intent = result.get("Refund Intent", "Yes" if result.get("refund_intent") == 1 else "No")
            prob = float(result.get("Refund Intent Probability", result.get("probability", 0.0)))
            model_threshold = float(result.get("threshold", 0.44))  # 从模型返回中动态获取
            key_features = result.get("key_features", [])

            st.markdown("---")
            st.subheader("Results")

            # 状态卡片与指标展示
            col_status, col_prob, col_thresh = st.columns([1.5, 1, 1])

            with col_status:
                if intent == "Yes":
                    st.error(f"🚨 Predicted Intent: **{intent} (Refund/Return)**")
                else:
                    st.success(f"💬 Predicted Intent: **{intent} (General Query)**")

            with col_prob:
                delta_val = prob - model_threshold
                st.metric(
                    label="Predicted Probability",
                    value=f"{prob:.2%}",
                    delta=f"{delta_val:+.2%} vs threshold"
                )

            with col_thresh:
                st.metric(
                    label="Model Optimal Threshold",
                    value=f"{model_threshold:.2f}",
                    help="Loaded dynamically from the trained model metadata."
                )

            st.progress(prob)

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
