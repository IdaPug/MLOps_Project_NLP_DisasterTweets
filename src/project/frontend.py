import os

import requests
import streamlit as st


def get_backend_url() -> str:
    """Get the BentoML backend URL."""
    backend = os.environ.get("BACKEND")

    if not backend:
        raise ValueError("BACKEND environment variable is not set")

    return backend.rstrip("/")


def predict_tweet(text: str, backend: str) -> dict | None:
    """Send a tweet to the BentoML backend."""
    response = requests.post(
        f"{backend}/predict",
        json={"text": text},
        timeout=30,
    )

    if response.status_code == 200:
        return response.json()

    return None


def main() -> None:
    """Run the Streamlit frontend."""
    st.set_page_config(
        page_title="Disaster Tweet Classifier",
        page_icon="☄️",
    )

    st.title("💥 Disaster Tweet Classifier")

    st.write("Enter a tweet below and the machine learning model " "will predict whether it is about a real disaster.")

    backend = get_backend_url()

    text = st.text_area(
        "Tweet",
        placeholder="e.g. There is a huge earthquake and people need help",
        height=150,
    )

    if st.button("Classify tweet", type="primary"):
        if not text.strip():
            st.warning("Please enter a tweet first.")
            return

        with st.spinner("Classifying..."):
            result = predict_tweet(text, backend)

        if result is None:
            st.error("Failed to get a prediction from the backend.")
            return

        st.subheader("Prediction")

        if result["prediction"] == 1:
            st.error("🚨 Disaster")
        else:
            st.success("✅ Not a disaster")

        st.write(f"**Confidence:** {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
