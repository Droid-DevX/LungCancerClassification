# Cancer Type Prediction using Deep Learning

This repository contains a deep learning-based cancer type prediction system using a trained convolutional neural network (CNN). The model is deployed using Streamlit, allowing users to upload medical images and receive predictions with a probability distribution displayed in a pie chart.

## Features

- Upload an image of a tissue sample.
- Get real-time cancer type predictions.
- Visual representation of prediction probabilities using a pie chart.
- Uses a trained CNN model (`Model.h5`).

## Cancer Types Detected

The model predicts five types of tissue conditions:

- **Colon Adenocarcinoma**
- **Colon Benign Tissue**
- **Lung Adenocarcinoma**
- **Lung Benign Tissue**
- **Lung Squamous Cell Carcinoma**

## Requirements

Ensure you have Python installed along with the following dependencies:

```bash
pip install streamlit tensorflow numpy matplotlib
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Droid-DevX/LungCancerClassification.git
   cd LungCancerClassification
   ```
2. Place the trained model file (`Model.h5`) in the project directory.
3. Run the application:
   ```bash
   streamlit run main.py
   ```
4. Open the provided local URL in your web browser.

## File Descriptions

- **`main.py`**: Contains the Streamlit-based web app for cancer type prediction.
- **`README.md`**: Documentation for setting up and using the project.
- **`lung_colon_image_set`**: Contains about 2% of the original data. You can download the full dataset from: [LC25000 Dataset](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af).

## Usage

1. Upload an image (JPG, PNG, JPEG).
2. The system will process the image and predict the cancer type probabilities.
3. The results will be displayed in a table along with a pie chart visualization.

## Example Output

- **Prediction Probabilities:**
  - Colon Adenocarcinoma: 90.0%
  - Colon Benign Tissue: 10.0%
  - Lung Adenocarcinoma: 0.0%
  - Lung Benign Tissue: 0.0%
  - Lung Squamous Cell Carcinoma: 0.0%

## License

This project is open-source. Feel free to modify and improve it!

## Acknowledgments

- TensorFlow for deep learning.
- Streamlit for interactive UI development.
- Medical image datasets used for training the model.

