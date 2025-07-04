import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import os


model_filename = 'xgb_tuned_model.pkl'
loaded_model = joblib.load(model_filename)

# Get the expected feature names from the trained model
expected_features = loaded_model.get_booster().feature_names


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def add_bg_from_local(image_path):
    base64_img = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


image_path = "/Users/dixonsmac/Desktop/Projects_2024/Landslide_Project/main_code/landslide_image.jpg"  # Replace with the path to your local image


add_bg_from_local(image_path)


st.title('Landslide Prediction App')

template_file_path = "template.csv"

# Check if file exists before displaying button
if os.path.exists(template_file_path):
    with open(template_file_path, "rb") as file:
        st.download_button(
            label="📥 Download Template",
            data=file,
            file_name="template.csv",
            mime="text/csv"
        )
else:
    st.error("⚠️ Template CSV file not found. Please check the file path.")
# Instructions
st.write("You can either manually input data or upload a .csv file with the required features.")

template_file_path = "template.csv"  # Ensure this file exists in your project root

try:
    with open(template_file_path, "rb") as file:
        st.download_button(
            label="Download Template CSV",
            data=file,
            file_name="template.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.error("Template CSV file not found. Please make sure 'template.csv' is in the project root.")


# Option to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file:
    # If a file is uploaded, load it into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(input_data)
else:
    # Manual input option
    st.subheader("Manual Input")
    aspect_NSEW = st.selectbox('Aspect NSEW', ['N', 'NE', 'NW', 'S', 'SE', 'SW', 'W'])
    elevation_center = st.number_input('Elevation Center', value=0.0)
    elevation_minmax_difference = st.number_input('Elevation Min-Max Difference', value=0.0)
    slope_center = st.number_input('Slope Center', value=0.0)
    slope_difference_pct = st.number_input('Slope Difference Percentage', value=0.0)
    aspect_span = st.number_input('Aspect Span', value=0.0)
    aspect_sum_span_9cells = st.number_input('Aspect Sum Span 9 Cells', value=0.0)
    placurv_center_below_diff = st.number_input('Plan Curvature Center Below Difference', value=0.0)
    placurv_signs_8 = st.selectbox('Plan Curvature Signs 8', ['++-', '+-+', '+--', '-++', '-+-', '--+', '---'])
    placurv_top_vs_2bottom = st.number_input('Plan Curvature Top vs 2 Bottom', value=0.0)
    procurv_signs_8 = st.selectbox('Profile Curvature Signs 8', ['++-', '+-+', '+--', '-++', '-+-', '--+', '---'])
    procurv_top_vs_2bottom = st.number_input('Profile Curvature Top vs 2 Bottom', value=0.0)
    lsfactor_sum_center_1below = st.number_input('LS Factor Sum Center 1 Below', value=0.0)
    twi_center = st.number_input('Topographic Wetness Index (TWI) Center', value=0.0)
    twi_25mean = st.number_input('TWI 25 Mean', value=0.0)
    geology_center = st.selectbox('Geology Center', [1, 2, 3, 4, 5])  # Example values
    geology_9mode = st.selectbox('Geology 9 Mode', [1, 2, 3, 4, 5])  # Example values
    sdoif_center = st.number_input('SDOIF Center', value=0.0)

    # Collect the inputs into a DataFrame
    input_data = pd.DataFrame({
        'aspect_NSEW': [aspect_NSEW],
        'elevation_center': [elevation_center],
        'elevation_minmax_difference': [elevation_minmax_difference],
        'slope_center': [slope_center],
        'slope_difference_pct': [slope_difference_pct],
        'aspect_span': [aspect_span],
        'aspect_sum_span_9cells': [aspect_sum_span_9cells],
        'placurv_center_below_diff': [placurv_center_below_diff],
        'placurv_signs_8': [placurv_signs_8],
        'placurv_top_vs_2bottom': [placurv_top_vs_2bottom],
        'procurv_signs_8': [procurv_signs_8],
        'procurv_top_vs_2bottom': [procurv_top_vs_2bottom],
        'lsfactor_sum_center_1below': [lsfactor_sum_center_1below],
        'twi_center': [twi_center],
        'twi_25mean': [twi_25mean],
        'geology_center': [geology_center],
        'geology_9mode': [geology_9mode],
        'sdoif_center': [sdoif_center]
    })

# Make predictions
if st.button('Predict'):
    # One-hot encode categorical features
    object_columns = input_data.select_dtypes(include=['object']).columns
    custom_input = pd.get_dummies(input_data, columns=object_columns, drop_first=True)

    # Align input features with the expected features
    for col in expected_features:
        if col not in custom_input:
            custom_input[col] = 0

    # Drop any extra columns not expected by the model
    custom_input = custom_input[expected_features]

    # Make predictions
    prediction = loaded_model.predict(custom_input)

    # Display results
    st.subheader("Prediction Results")
    if len(prediction) == 1:
        st.write(f"Predicted Class: {'Landslide' if prediction[0] == 1 else 'No Landslide'}")
    else:
        input_data['Prediction'] = ['Landslide' if p == 1 else 'No Landslide' for p in prediction]
        st.write(input_data[['Prediction']])

# Display the input DataFrame for reference
st.write("Input data used for prediction:")
st.dataframe(input_data)

