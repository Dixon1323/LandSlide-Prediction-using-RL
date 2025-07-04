import streamlit as st
import subprocess
import platform
import os
import base64

# Function to open a new terminal and run the Streamlit app
def run_app(app_name):
    current_os = platform.system()
    if current_os == 'Windows':
        command = f'start cmd /k "source landslide/bin/activate && streamlit run {app_name}"'
        subprocess.run(command, shell=True)
    elif current_os == 'Darwin':  # macOS
        command = f'osascript -e \'tell application "Terminal" to do script "source /Users/dixonsmac/Desktop/projects_2024/Landslide_Project/main_code/landslide/bin/activate && streamlit run {app_name}"\''
        subprocess.run(command, shell=True)
    else:  # Linux
        command = f'gnome-terminal -- bash -c "source activate your_env_name && streamlit run {app_name}; exec bash"'
        subprocess.run(command, shell=True)

# Function to encode local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Add background image using base64
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

# Set background image
image_path = "home_page.jpg"  # Replace with the correct image path
add_bg_from_local(image_path)

# Streamlit Homepage UI
st.title("Landslide and Object Distance Prediction Application")
st.write("Welcome to the Landslide Prediction System. This tool helps predict landslides and measures object distances in affected areas.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Contact Us"])

if page == "Home":
    st.header("Overview")
    st.write("""TLandslides are among the most dangerous natural disasters, causing severe damage to infrastructure, livelihoods, and human lives. To mitigate risks and enhance preparedness, our Landslide and Object Distance Prediction Application provides an AI-powered solution for real-time monitoring and early warning.
    This application leverages machine learning and computer vision techniques to predict landslides based on terrain conditions, environmental parameters, and visual data. Additionally, it features an object distance prediction module, which helps assess the movement of objects within landslide-prone zones, improving safety and response planning.""")
    st.write("Choose one of the following options:")

# Create two buttons for each option
    if st.button("Landslide Prediction"):
    # st.write("Opening Landslide Prediction App...")
        run_app('/Users/dixonsmac/Desktop/projects_2024/Landslide_Project/main_code/app_landslide.py')

    if st.button("Landslide Object Distance Prediction"):
    # st.write("Opening Landslide Object Distance Prediction App...")
        run_app('/Users/dixonsmac/Desktop/projects_2024/Landslide_Project/main_code/app_distance.py')


elif page == "Contact Us":
    st.header("Contact Us")
    st.write("For any inquiries or support, feel free to contact us:")
    st.write("📧 Email: support@landslidepredict.com")
    st.write("📞 Phone: +123-456-7890")
    st.write("🌐 Website: [LandslidePredict.com](https://terrainanomalydetector.pythonanywhere.com)")
