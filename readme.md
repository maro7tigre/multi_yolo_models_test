
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
