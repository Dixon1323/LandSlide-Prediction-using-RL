import streamlit as st
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from manim import *
import base64
import time

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

# Example: Add your own local image path
image_path = "image.png"  # Replace with the path to your local image

# Add background image to Streamlit app
add_bg_from_local(image_path)


placeholder = st.empty()


total_distance=0


# Reinforcement Learning Environment
class LandslideEnv(gym.Env):
    def __init__(self, slope_angle=30, truck_weight=40000, debris_density=2000, debris_volume=500):
        super(LandslideEnv, self).__init__()

        # Store the custom parameters
        self.slope_angle = slope_angle
        self.truck_weight = truck_weight
        self.debris_density = debris_density
        self.debris_volume = debris_volume

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Initial positions
        self.initial_truck_position = np.array([5.0, -3.0], dtype=np.float32)
        self.initial_debris_position = np.array([0.0, 0.0], dtype=np.float32)

        # Constants
        self.gravity = 9.8  # m/s^2
        self.debris_mass = self.debris_density * self.debris_volume  # kg
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.truck_position = np.copy(self.initial_truck_position)
        self.debris_position = np.copy(self.initial_debris_position)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.truck_position, self.debris_position]).astype(np.float32)

    def step(self, action):
        old_truck_position = np.copy(self.truck_position)
        force_x, force_y = action
        self.truck_position += np.array([force_x, force_y], dtype=np.float32)
        self.debris_position += np.array([0.1, -0.1], dtype=np.float32)

        distance_moved = np.linalg.norm(self.truck_position - old_truck_position)

        truck_friction_force = self.truck_weight * self.gravity * np.cos(np.radians(self.slope_angle)) * 0.5  # N
        debris_effective_force = self.debris_mass * 10 * np.sin(np.radians(self.slope_angle))  # Simplified

        if debris_effective_force > truck_friction_force:
            reward = 1
        else:
            reward = -1

        terminated = bool(self.debris_position[1] <= -5)
        truncated = False
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

# Streamlit app integration
st.title("Landslide Simulation with Reinforcement Learning")

# User input for simulation parameters
slope_angle = st.slider("Slope Angle (degrees)", 0, 90, 30)
truck_weight = st.number_input("Truck Weight (kg)", 1000, 100000, 40000)
debris_density = st.number_input("Debris Density (kg/m³)", 1000, 10000, 2000)
debris_volume = st.number_input("Debris Volume (m³)", 100, 20000, 500)

# Create the environment with user-defined parameters
custom_env = LandslideEnv(slope_angle=slope_angle, truck_weight=truck_weight, debris_density=debris_density, debris_volume=debris_volume)

# Check the custom environment
check_env(custom_env)

# Wrap the environment
custom_env = Monitor(custom_env)
custom_env = DummyVecEnv([lambda: custom_env])

#pre load the model if available

model = PPO.load("ppo_landslide_custom")





# Run the Manim simulation
class LandslideSimulation(Scene):
    def construct(self):
        text1 = "Evaluvating.."
        styled_text1 = f'<p style="background-color:red; justify-content: center; align-items: center; color:black; font-size:20px;">{text1}</p>'
        placeholder.markdown(styled_text1, unsafe_allow_html=True)
        slope = Line(start=ORIGIN, end=RIGHT * 5 + DOWN * 3, color=GRAY)
        self.play(Create(slope))

        debris_size = 0.1
        debris_particles = VGroup(*[
            Dot(point=[i / 2, -i / 3, 0], color=RED).scale(debris_size) for i in range(0, 11)
        ])
        self.play(Create(debris_particles))

        truck = Square(side_length=0.5, color=BLUE).move_to([0, 0, 0])
        truck_label = Text("Truck", font_size=24).next_to(truck, UP)
        truck_with_label = VGroup(truck, truck_label)
        self.play(Create(truck_with_label))

        obs = custom_env.reset()
        total_distance = 0
        initial_position = obs[0, :2]

        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = custom_env.step(action)
            truck_position = obs[0, :2]

            distance_moved = np.linalg.norm(truck_position - initial_position)
            total_distance += distance_moved

            self.play(truck_with_label.animate.move_to([truck_position[0], truck_position[1], 0]), run_time=0.1)

            for dot in debris_particles:
                x, y, _ = dot.get_center()
                x += 0.05
                y -= 0.05
                if y > -3.0:
                    dot.move_to([x, y, 0])

            initial_position = truck_position

            if done:
                break
        text1 = "Evaluvation Complete !"
        styled_text1 = f'<p style="background-color:green; justify-content: center; align-items: center; color:black; font-size:20px;">{text1}</p>'
        placeholder.markdown(styled_text1, unsafe_allow_html=True)
        print(f"Total distance traveled by the truck: {total_distance:.2f} meters")
        result_text = Text(f"Truck traveled {total_distance:.2f} meters")
        st.write(f"Truck traveled {total_distance:.2f} meters")
        eval=True
        result_text.to_edge(DOWN)
        self.play(Write(result_text))
        self.wait()


if st.button("Train Model"):
    print('*'*90)
    with st.spinner("Training the model..."):
        # Define and train the DRL model
        model = PPO('MlpPolicy', custom_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01)
        model.learn(total_timesteps=100000)
        model.save("ppo_landslide_custom")
        model = PPO.load("ppo_landslide_custom")
        print(f"slope angle = {slope_angle}, truck_weight = {truck_weight}, debris_density = {debris_density}, debris_volume={debris_volume}")
    st.success("Training complete!")
        


# Evaluate the model
if st.button("Evaluate Model"):
    print('*'*90)
    mean_reward, std_reward = evaluate_policy(model, custom_env, n_eval_episodes=10)
    scene = LandslideSimulation()
    scene.render()
    time.sleep(2)
    video_path = r"media/videos/1080p60/LandslideSimulation.mp4"
    st.video(video_path)