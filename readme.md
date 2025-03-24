
## 1. Create a Virtual Environment

Open your terminal in the project directory and run:

```bash
python -m venv .venv
```

## 2. Activate the Virtual Environment

- **On Windows:**
  ```bash
  .venv\Scripts\activate
  ```

- **On macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

## 3. Install Required Packages

With the virtual environment activated, install the dependencies:

```bash
pip install ultralytics opencv-python pillow numpy
pip install torch torchvision  # For PyTorch models
pip install onnxruntime  # For ONNX models
pip install tensorflow  # For TF/Keras models
```

## 4. Project Structure
- `main_multiplatforms.py`: The script to run the multi platforms application.
- `complete_dataset/`: Contains dataset provided by [trashNet](https://github.com/garythung/trashnet.git)
- `helper_split_train_val.py`: Scripts to split and resize(640x640) the dataset into train and validation sets(you need to extract `complete_dataset` first).
- `train_classifier.py`: Script to train the classifier(you need to split train/val first).

### main app :
- `main_cam.py`: The main script to run the application for camera input.
- `main_img.py`: The main script to run the application for images input.
- `core.py`: The core script required for the main scripts...
- `multimodel_llm.py`: The script to run the LLMs experimenting application.

## 5. LLM experimentation

Create a `credentials.py` file in the root directory of the project with the following content:

```python

# API credentials
API_KEY = "YOUR_API_KEY"

# Function to get API key
def get_api_key():
    return API_KEY

```

Make sure to:
3. Create an API key from [Gemini-API](https://console.cloud.google.com/apis/)
2. Replace the placeholder values with your actual API keys


## 6. Aknowledgements

the `dataset/` and `complete_dataset` directory is provided by [trashNet](https://github.com/garythung/trashnet.git) under MIT License.

