import gradio as gr
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load models
with open('best_model_fire_occurrence.pkl', 'rb') as f:
    model_fire_occurrence = pickle.load(f)

with open('best_model_suppression_cost.pkl', 'rb') as f:
    model_suppression_cost = pickle.load(f)

with open('best_model_fire_duration.pkl', 'rb') as f:
    model_fire_duration = pickle.load(f)

with open('best_model_fire_size.pkl', 'rb') as f:
    model_fire_size = pickle.load(f)

# Load the pre-fitted scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict(temp, humidity, wind_speed, rainfall, fuel_moisture, vegetation_type, slope, region):
    # Validate inputs
    if vegetation_type is None or region is None:
        return "Please select valid options for vegetation type and region."

    # Convert categorical variables into numerical format
    vegetation_type_encoded = {"Grassland": 0, "Forest": 1, "Shrubland": 2}.get(vegetation_type, -1)
    region_encoded = {"North": 0, "South": 1, "East": 2, "West": 3}.get(region, -1)

    # If either encoded variable is -1, return an error
    if vegetation_type_encoded == -1 or region_encoded == -1:
        return "Invalid selection for vegetation type or region."

    # Prepare the input array
    input_array = np.array([[temp, humidity, wind_speed, rainfall, fuel_moisture, vegetation_type_encoded, slope, region_encoded]])

    # Pad the input array to match the expected number of features (11 in this case)
    input_array_padded = np.hstack([input_array, np.zeros((1, 3))])  # Adding three zeros

    # Scale input values using the loaded, pre-fitted scaler
    input_scaled = scaler.transform(input_array_padded)

    # Make predictions
    fire_occurrence = model_fire_occurrence.predict(input_scaled)[0]
    suppression_cost = model_suppression_cost.predict(input_scaled)[0]
    fire_duration = model_fire_duration.predict(input_scaled)[0]
    fire_size = model_fire_size.predict(input_scaled)[0]

    # Convert fire_occurrence to "Yes" or "No"
    if fire_occurrence == 0:
      fire_occurrence_display = "Yes"
    else:
      fire_occurrence_display = "No"

    # Convert predictions to absolute values and round to 0 decimal places
    suppression_cost = round((float(suppression_cost)), 0)
    fire_duration = round(abs(float(fire_duration)), 0)
    fire_size = round((float(fire_size)), 0)

    return fire_occurrence_display, suppression_cost, fire_duration, fire_size

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Number(label="Temperature (°C)"),
        gr.components.Number(label="Humidity (%)"),
        gr.components.Number(label="Wind Speed (km/h)"),
        gr.components.Number(label="Rainfall (mm)"),
        gr.components.Number(label="Fuel Moisture (%)"),
        gr.components.Dropdown(choices=["Grassland", "Forest", "Shrubland"], label="Vegetation Type", value="Grassland"),
        gr.components.Number(label="Slope (%)"),
        gr.components.Dropdown(choices=["North", "South", "East", "West"], label="Region", value="North"),
    ],
import gradio as gr
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load models
with open('best_model_fire_occurrence.pkl', 'rb') as f:
    model_fire_occurrence = pickle.load(f)

with open('best_model_suppression_cost.pkl', 'rb') as f:
    model_suppression_cost = pickle.load(f)

with open('best_model_fire_duration.pkl', 'rb') as f:
    model_fire_duration = pickle.load(f)

with open('best_model_fire_size.pkl', 'rb') as f:
    model_fire_size = pickle.load(f)

# Load the pre-fitted scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict(temp, humidity, wind_speed, rainfall, fuel_moisture, vegetation_type, slope, region):
    # Validate inputs
    if vegetation_type is None or region is None:
        return "Please select valid options for vegetation type and region."

    # Convert categorical variables into numerical format
    vegetation_type_encoded = {"Grassland": 0, "Forest": 1, "Shrubland": 2}.get(vegetation_type, -1)
    region_encoded = {"North": 0, "South": 1, "East": 2, "West": 3}.get(region, -1)

    # If either encoded variable is -1, return an error
    if vegetation_type_encoded == -1 or region_encoded == -1:
        return "Invalid selection for vegetation type or region."

    # Prepare the input array
    input_array = np.array([[temp, humidity, wind_speed, rainfall, fuel_moisture, vegetation_type_encoded, slope, region_encoded]])

    # Pad the input array to match the expected number of features (11 in this case)
    input_array_padded = np.hstack([input_array, np.zeros((1, 3))])  # Adding three zeros

    # Scale input values using the loaded, pre-fitted scaler
    input_scaled = scaler.transform(input_array_padded)

    # Make predictions
    fire_occurrence = model_fire_occurrence.predict(input_scaled)[0]
    suppression_cost = model_suppression_cost.predict(input_scaled)[0]
    fire_duration = model_fire_duration.predict(input_scaled)[0]
    fire_size = model_fire_size.predict(input_scaled)[0]

    # Convert fire_occurrence to "Yes" or "No"
    if fire_occurrence == 0:
      fire_occurrence_display = "Yes"
    else
      fire_occurrence_display = "No"

    # Convert predictions to absolute values and round to 0 decimal places
    suppression_cost = round((float(suppression_cost)), 0)
    fire_duration = round(abs(float(fire_duration)), 0)
    fire_size = round((float(fire_size)), 0)

    return fire_occurrence_display, suppression_cost, fire_duration, fire_size

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Number(label="Temperature (°C)"),
        gr.components.Number(label="Humidity (%)"),
        gr.components.Number(label="Wind Speed (km/h)"),
        gr.components.Number(label="Rainfall (mm)"),
        gr.components.Number(label="Fuel Moisture (%)"),
        gr.components.Dropdown(choices=["Grassland", "Forest", "Shrubland"], label="Vegetation Type", value="Grassland"),
        gr.components.Number(label="Slope (%)"),
        gr.components.Dropdown(choices=["North", "South", "East", "West"], label="Region", value="North"),
    ],
    outputs=[
        gr.components.Label(label="Fire Occurrence"),
        gr.components.Label(label="Suppression Cost ($)"),
        gr.components.Label(label="Fire Duration (hrs)"),
        gr.components.Label(label="Fire Size (hectares)"),
    ],
    title="Fire Prediction Model",
    description="Enter the input variables to predict fire occurrence, suppression cost, fire duration, and fire size."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
