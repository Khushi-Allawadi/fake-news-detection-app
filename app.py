import streamlit as st
from backend import run_app

def main():
    st.title("Fake News Detection App")
    results = run_app()

    st.write("Accuracy of Classifiers:")
    st.write("- Random Forest: {:.2f}%".format(results['rf_accuracy'] * 100))
    st.write("- SVM: {:.2f}%".format(results['svm_accuracy'] * 100))
    st.write("- Logistic Regression: {:.2f}%".format(results['lr_accuracy'] * 100))
    st.write("- Gradient Boosting: {:.2f}%".format(results['gb_accuracy'] * 100))

    title_input = st.text_input("Enter the title of the article:")
    text_input = st.text_area("Enter the text of the article:")

    classifier = st.selectbox("Choose Classifier", ["Random Forest", "SVM", "Logistic Regression", "Gradient Boosting"])

    if st.button("Classify"):
        prediction = None
        if classifier == "Random Forest":
            prediction = results['rf_classifier'].predict_article(title_input, text_input)
        elif classifier == "SVM":
            prediction = results['svm_classifier'].predict_article(title_input, text_input)
        elif classifier == "Logistic Regression":
            prediction = results['lr_classifier'].predict_article(title_input, text_input)
        elif classifier == "Gradient Boosting":
            prediction = results['gb_classifier'].predict_article(title_input, text_input)

        if prediction == 1:
            st.error("Fake News")
        else:
            st.success("Real News")

if __name__ == "__main__":
    main()
