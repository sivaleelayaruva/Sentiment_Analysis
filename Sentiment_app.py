import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 1. Load model and vectorizer
# Ensure these files are in the same folder as this script
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found! Please ensure model.pkl and vectorizer.pkl are in the folder.")

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard")

# 2. Sidebar
st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ["Single Text", "CSV Upload"])

# ------------------ MODE: SINGLE TEXT ------------------
if mode == "Single Text":
    st.subheader("Analyze Single Text")

    user_input = st.text_area("Enter text:")

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter text")
        else:
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]
            prob = model.predict_proba(X).max()

            if pred == "positive":
                st.success(f"Positive 😊 ({prob:.2f})")
            elif pred == "negative":
                st.error(f"Negative 😡 ({prob:.2f})")
            else:
                st.info(f"Neutral 😐 ({prob:.2f})")

# ------------------ MODE: CSV UPLOAD ------------------
elif mode == "CSV Upload":
    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Preview Data")
        st.dataframe(df.head())

        # Select column
        text_column = st.selectbox("Select Text Column", df.columns)

        if st.button("Analyze CSV"):
            texts = df[text_column].astype(str)

            # Process predictions
            X = vectorizer.transform(texts)
            predictions = model.predict(X)
            df["Sentiment"] = predictions

            st.write("### Results")
            st.dataframe(df)

            # 📊 DASHBOARD SECTION
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Sentiment Distribution")
                sentiment_counts = df["Sentiment"].value_counts()
                fig, ax = plt.subplots()
                ax.bar(sentiment_counts.index, sentiment_counts.values, color=['#4CAF50', '#F44336', '#2196F3'])
                st.pyplot(fig)

            with col2:
                st.write("### Percentage Split")
                fig2, ax2 = plt.subplots()
                ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                st.pyplot(fig2)

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Analyzed Data",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )