import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub




new_model = tf.keras.models.load_model("Fake news nlp model.h5",custom_objects={"KerasLayer": hub.KerasLayer})





def welcome():
    return "Welcome to my app"


def main():
    st.title("Fake News Detection App")
    st.write(
        "This app will tell you if mention news is Fake or Real by using Natural Language Processing")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Fake News Detector </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    text = st.text_input("Enter your News")


    if st.button("Predict"):
        pred_prob = new_model.predict([text])
        predict = tf.squeeze(tf.round(pred_prob)).numpy()
        st.subheader("AI thinks that ...")

        if predict > 0:

            st.success(
                f"It's Real news, you can trust it. Confidence Level is {tf.round(pred_prob,3)*100}%",icon="✅")
        else:
            st.warning(
                f"Beware!! It's a Fake News. Confidence Level is {tf.round(100 - pred_prob,2)}%", icon="⚠️")

    if st.button("About"):

        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()

