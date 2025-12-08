import streamlit as st
import joblib

# Load trained model
model = joblib.load("cuisine_classifier.pkl")

# ---- UI Styling ----
st.set_page_config(page_title="Cuisine Predictor", page_icon="🍽️")

st.title("🍽️ Name That Cuisine!")
st.write("Enter your recipe ingredients and let the AI guess the cuisine.")

st.markdown("---")

# ---- User Input ----
ingredients = st.text_area(
    "🧾 Enter ingredients (comma or space separated):",
    placeholder="Example: tomato, garlic, basil, olive oil, pasta"
)

# ---- Cuisine Emojis ----
cuisine_emojis = {
    "italian": "🍝",
    "mexican": "🌮",
    "chinese": "🥡",
    "japanese": "🍣",
    "indian": "🍛",
    "thai": "🍜",
    "french": "🥐",
    "greek": "🥙",
    "korean": "🍱",
    "spanish": "🥘",
    "british": "🍵",
    "vietnamese": "🍲",
    "moroccan": "🧆",
    "russian": "🥟",
    "irish": "🥔"
}

# ---- Predict Button ----
if st.button("Predict Cuisine"):
    if not ingredients.strip():
        st.warning("⚠️ Please enter at least one ingredient.")
    else:
        prediction = model.predict([ingredients])[0]
        proba = model.predict_proba([ingredients])[0]
        confidence = max(proba) * 100

        emoji = cuisine_emojis.get(prediction.lower(), "🍽️")

        st.success(f"{emoji} **Predicted Cuisine: {prediction.upper()}**")
        st.write(f"📊 Confidence: **{confidence:.2f}%**")

        if confidence < 40:
            st.info("🤔 The model is unsure — this might be a fusion dish or uncommon combo.")
        elif confidence > 80:
            st.balloons()
