import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import threading
import time
import os
import importlib
from PIL import Image, ImageTk

# MARK: Model Manager
class ModelManager:
    """Handles loading and inference for various model types"""
    
    def __init__(self):
        # Check available frameworks
        self.frameworks = self._check_available_frameworks()
        print(f"Available frameworks: {self.frameworks}")
        
        # Initialize model categories and formats
        self.model_formats = {
            "pt": {"name": "PyTorch", "supported": "torch" in self.frameworks},
            "onnx": {"name": "ONNX", "supported": "onnxruntime" in self.frameworks},
            "h5": {"name": "TensorFlow/Keras", "supported": "tensorflow" in self.frameworks},
            "tflite": {"name": "TensorFlow Lite", "supported": "tensorflow" in self.frameworks},
            "pb": {"name": "TensorFlow SavedModel", "supported": "tensorflow" in self.frameworks},
        }
        
        # Scan for available models
        self.available_models = self._scan_models()
        
        # Store loaded model instances
        self.model_instances = {}
        
        # Standard YOLOv8 models (for reference)
        self.standard_yolo_models = [
            "yolov8n.pt", 
            "yolov8s.pt", 
            "yolov8m.pt", 
            "yolov8l.pt", 
            "yolov8x.pt"
        ]
    
    def _check_available_frameworks(self):
        """Check which ML frameworks are available"""
        frameworks = []
        
        # Check for PyTorch
        try:
            importlib.import_module("torch")
            frameworks.append("torch")
        except ImportError:
            pass
        
        # Check for TensorFlow
        try:
            importlib.import_module("tensorflow")
            frameworks.append("tensorflow")
        except ImportError:
            pass
        
        # Check for ONNX Runtime
        try:
            importlib.import_module("onnxruntime")
            frameworks.append("onnxruntime")
        except ImportError:
            pass
        
        # Check for Ultralytics YOLO
        try:
            importlib.import_module("ultralytics")
            frameworks.append("ultralytics")
        except ImportError:
            pass
        
        return frameworks
    
    def _scan_models(self):
        """Scan for available model files and categorize them"""
        models = {}
        for file in os.listdir('.'):
            ext = file.split('.')[-1].lower()
            if ext in self.model_formats:
                if self.model_formats[ext]["supported"]:
                    if ext not in models:
                        models[ext] = []
                    models[ext].append(file)
        return models
    
    def get_all_model_files(self):
        """Get all available model files"""
        all_models = []
        for model_type in self.available_models.values():
            all_models.extend(model_type)
        return all_models
    
    def load_model(self, model_name):
        """Load a model based on its file extension"""
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        
        # Extract file extension
        ext = model_name.split('.')[-1].lower()
        
        if ext not in self.model_formats or not self.model_formats[ext]["supported"]:
            raise ValueError(f"Unsupported model format: {ext}")
        
        try:
            # Load model based on extension
            if ext == "pt":
                model = self._load_pytorch_model(model_name)
            elif ext == "onnx":
                model = self._load_onnx_model(model_name)
            elif ext in ["h5", "pb"]:
                model = self._load_tensorflow_model(model_name)
            elif ext == "tflite":
                model = self._load_tflite_model(model_name)
            else:
                raise ValueError(f"Unsupported model format: {ext}")
            
            # Store the loaded model
            self.model_instances[model_name] = model
            return model
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_pytorch_model(self, model_name):
        """Load a PyTorch model"""
        try:
            # Check if it's a YOLO model
            if "yolo" in model_name.lower() and "ultralytics" in self.frameworks:
                from ultralytics import YOLO
                return {"type": "yolo", "model": YOLO(model_name)}
            else:
                import torch # type: ignore
                model = torch.load(model_name, map_location=torch.device('cpu'))
                if hasattr(model, 'eval'):
                    model.eval()
                return {"type": "pytorch", "model": model}
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            raise
    
    def _load_onnx_model(self, model_name):
        """Load an ONNX model"""
        try:
            import onnxruntime as ort # type: ignore
            session = ort.InferenceSession(model_name)
            
            # Get model input details
            model_inputs = session.get_inputs()
            input_names = [input.name for input in model_inputs]
            input_shapes = [input.shape for input in model_inputs]
            
            return {
                "type": "onnx",
                "model": session,
                "input_names": input_names,
                "input_shapes": input_shapes
            }
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
    
    def _load_tensorflow_model(self, model_name):
        """Load a TensorFlow/Keras model"""
        try:
            import tensorflow as tf  # type: ignore
            
            # Check if it's a SavedModel directory or .h5 file
            if model_name.endswith('.pb') or os.path.isdir(model_name):
                model = tf.saved_model.load(model_name)
                model_type = "tf_saved_model"
            else:  # .h5 file
                model = tf.keras.models.load_model(model_name)
                model_type = "keras"
            
            return {"type": model_type, "model": model}
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            raise
    
    def _load_tflite_model(self, model_name):
        """Load a TensorFlow Lite model"""
        try:
            import tensorflow as tf # type: ignore
            
            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_name)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            return {
                "type": "tflite",
                "model": interpreter,
                "input_details": input_details,
                "output_details": output_details
            }
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            raise
    
    def predict(self, frame, model_name):
        """Run inference on a frame using the specified model"""
        if model_name not in self.model_instances:
            model = self.load_model(model_name)
            if model is None:
                return frame, None
        else:
            model = self.model_instances[model_name]
        
        model_type = model["type"]
        
        try:
            # Different inference based on model type
            if model_type == "yolo":
                results = model["model"](frame)
                annotated_frame = results[0].plot()
                return annotated_frame, results[0]
            
            elif model_type == "pytorch":
                # Basic PyTorch inference (needs customization for specific models)
                import torch # type: ignore
                import torchvision.transforms as transforms # type: ignore
                
                # Preprocess: Convert to tensor and normalize
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])
                
                # Convert frame to RGB and apply transform
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_img = transform(Image.fromarray(rgb_frame)).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    outputs = model["model"](tensor_img)
                
                # For visualization, just return the original frame
                # This should be customized for actual model output visualization
                return frame, outputs
            
            elif model_type == "onnx":
                # ONNX Runtime inference
                # Preprocess frame for the ONNX model
                input_name = model["input_names"][0]
                input_shape = model["input_shapes"][0]
                
                # Resize image to match model input
                if len(input_shape) == 4:  # Batch, Height, Width, Channels
                    height, width = input_shape[1:3]
                    if height > 0 and width > 0:
                        resized = cv2.resize(frame, (width, height))
                    else:
                        resized = frame
                else:
                    resized = frame
                
                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values
                img_data = rgb_img.astype(np.float32) / 255.0
                
                # Add batch dimension
                img_data = np.expand_dims(img_data, 0)
                
                # Run inference
                outputs = model["model"].run(None, {input_name: img_data})
                
                # For visualization, just return the original frame
                # This should be customized for actual model output visualization
                return frame, outputs
            
            elif model_type in ["keras", "tf_saved_model"]:
                # TensorFlow model inference
                import tensorflow as tf # type: ignore
                
                # Preprocess frame (assuming standard image input)
                img = cv2.resize(frame, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, 0)  # Add batch dimension
                
                # Run inference
                if model_type == "keras":
                    preds = model["model"].predict(img)
                else:  # tf_saved_model
                    infer = model["model"].signatures["serving_default"]
                    preds = infer(tf.constant(img))
                
                # For visualization, just return the original frame
                # This should be customized for actual model output visualization
                return frame, preds
            
            elif model_type == "tflite":
                # TensorFlow Lite inference
                interpreter = model["model"]
                input_details = model["input_details"]
                output_details = model["output_details"]
                
                # Preprocess frame
                input_shape = input_details[0]['shape']
                if len(input_shape) == 4:  # Batch, Height, Width, Channels
                    height, width = input_shape[1:3]
                    if height > 0 and width > 0:
                        img = cv2.resize(frame, (width, height))
                    else:
                        img = frame
                else:
                    img = frame
                
                # Further preprocessing depends on the specific model
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, 0)  # Add batch dimension
                
                # Set the input tensor
                interpreter.set_tensor(input_details[0]['index'], img)
                
                # Run inference
                interpreter.invoke()
                
                # Get the output tensors
                outputs = []
                for output_detail in output_details:
                    output = interpreter.get_tensor(output_detail['index'])
                    outputs.append(output)
                
                # For visualization, just return the original frame
                # This should be customized for actual model output visualization
                return frame, outputs
            
            else:
                return frame, None
                
        except Exception as e:
            print(f"Error during inference with {model_name}: {e}")
            # Return original frame if inference fails
            return frame, None


# MARK: Camera Manager
class CameraManager:
    """Handles camera detection and streaming"""
    
    def __init__(self):
        self.available_cameras = self._detect_cameras()
        self.current_camera = None
        self.current_camera_index = None
    
    def _detect_cameras(self):
        """Detect available cameras on the system"""
        available_cameras = {}
        # Try more camera indices (0 to 10) and be more lenient about detection
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to read a frame, but don't require success
                    ret, frame = cap.read()
                    # Even if we can't read a frame, if the camera opened, list it
                    available_cameras[f"Camera {i}"] = i
                cap.release()
            except Exception as e:
                print(f"Error detecting camera {i}: {e}")
        
        # If no cameras detected, add at least camera 0 as a fallback
        if not available_cameras and os.name == 'nt':  # Windows
            available_cameras["Camera 0"] = 0
        
        return available_cameras
    
    def open_camera(self, camera_idx):
        """Open a camera by index"""
        if self.current_camera is not None:
            self.current_camera.release()
        
        self.current_camera = cv2.VideoCapture(camera_idx)
        if self.current_camera.isOpened():
            self.current_camera_index = camera_idx
            return True
        else:
            self.current_camera = None
            self.current_camera_index = None
            return False
    
    def read_frame(self):
        """Read a frame from the current camera"""
        if self.current_camera is None:
            return False, None
        
        return self.current_camera.read()
    
    def release(self):
        """Release the current camera"""
        if self.current_camera is not None:
            self.current_camera.release()
            self.current_camera = None
            self.current_camera_index = None


# MARK: Display Panel
class DisplayPanel:
    """A panel that can display either raw camera feed or processed model output"""
    
    def __init__(self, parent, panel_id, models):
        self.parent = parent
        self.panel_id = panel_id
        self.models = models
        
        # Create UI elements
        self.frame = ttk.LabelFrame(parent, text=f"Display {panel_id}")
        self.canvas = tk.Canvas(self.frame, bg="black", width=320, height=240)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls
        controls_frame = ttk.Frame(self.frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Display mode selection
        ttk.Label(controls_frame, text="Display:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Create options: "None", "Camera Feed", and all models
        options = ["None", "Camera Feed"] + models
        
        self.display_var = tk.StringVar(value="None")
        self.display_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.display_var,
            values=options,
            state="readonly",
            width=15
        )
        self.display_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Last displayed image
        self.photo_img = None
    
    def update_models(self, models):
        """Update the list of available models"""
        options = ["None", "Camera Feed"] + models
        self.display_combo['values'] = options
    
    def update_frame(self, camera_frame, model_manager):
        """Update the display with a new frame"""
        display_type = self.display_var.get()
        
        if display_type == "None":
            # Clear display
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No Display",
                fill="white",
                font=('Arial', 14)
            )
            return
        
        elif display_type == "Camera Feed":
            # Display raw camera feed
            display_frame = camera_frame
        
        else:
            # Display model processed feed
            model_name = display_type
            try:
                # Process with the selected model
                display_frame, _ = model_manager.predict(camera_frame, model_name)
            except Exception as e:
                # If model processing fails, show an error
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2,
                    text=f"Error: {str(e)}",
                    fill="red",
                    font=('Arial', 12),
                    width=self.canvas.winfo_width() - 20
                )
                return
        
        # Convert frame to Tkinter PhotoImage
        self.photo_img = self._convert_frame_to_tk(display_frame)
        
        if self.photo_img:
            # Display the image
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self.photo_img
            )
    
    def _convert_frame_to_tk(self, frame):
        """Convert OpenCV frame to Tkinter PhotoImage"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use default dimensions if canvas not yet sized
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 320
            canvas_height = 240
        
        # Calculate scale to fit canvas while maintaining aspect ratio
        scale = min(
            canvas_width / frame.shape[1],
            canvas_height / frame.shape[0]
        )
        
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        
        if width > 0 and height > 0:
            # Resize and convert the frame
            resized = cv2.resize(frame, (width, height))
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            return ImageTk.PhotoImage(image=img)
        
        return None


# MARK: Model Selection Dialog
class ModelSelectionDialog:
    """Dialog for selecting and configuring models"""
    
    def __init__(self, parent, model_manager, callback):
        self.parent = parent
        self.model_manager = model_manager
        self.callback = callback
        self.dialog = None
        self.model_vars = {}
        self.installing = False
    
    def show(self):
        """Show the model selection dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Install Models")
        self.dialog.geometry("600x400")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Main frame
        frame = ttk.Frame(self.dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different model types
        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Initialize model variables
        self.model_vars = {}
        
        # Create tabs for each supported model type
        self._create_model_tabs()
        
        # Button frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Refresh button
        ttk.Button(
            button_frame,
            text="Refresh Models",
            command=self._refresh_models
        ).pack(side=tk.LEFT, padx=5)
        
        # Apply button
        ttk.Button(
            button_frame,
            text="Apply",
            command=self._apply_selection
        ).pack(side=tk.RIGHT, padx=5)
        
        # Cancel button
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def _create_model_tabs(self):
        """Create the tabs for each model type"""
        # Clear existing tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        
        # Create tabs for each supported model type
        for ext, formats in self.model_manager.model_formats.items():
            if formats["supported"]:
                tab = ttk.Frame(self.notebook, padding=10)
                self.notebook.add(tab, text=formats["name"])
                
                # Create scrollable frame for models
                scroll_frame = ttk.Frame(tab)
                scroll_frame.pack(fill=tk.BOTH, expand=True)
                
                # Add scrollbar
                scrollbar = ttk.Scrollbar(scroll_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Create canvas for scrolling
                canvas = tk.Canvas(scroll_frame, yscrollcommand=scrollbar.set)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Configure scrollbar
                scrollbar.config(command=canvas.yview)
                
                # Create inner frame for models
                inner_frame = ttk.Frame(canvas)
                canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
                
                # Add models
                models = self.model_manager.available_models.get(ext, [])
                
                if not models:
                    ttk.Label(
                        inner_frame, 
                        text=f"No {formats['name']} models found.\nPlace models in the application directory.",
                        foreground="gray"
                    ).pack(pady=10)
                
                # Add existing models with checkboxes
                if models:
                    ttk.Label(
                        inner_frame,
                        text="Available Models:",
                        font=("Arial", 10, "bold")
                    ).pack(anchor=tk.W, pady=(0, 5))
                
                for model in models:
                    var = tk.BooleanVar(value=True)  # Selected by default
                    
                    cb = ttk.Checkbutton(
                        inner_frame,
                        text=model,
                        variable=var
                    )
                    cb.pack(anchor=tk.W, padx=5, pady=2)
                    
                    self.model_vars[model] = var
                
                # Add predefined models section for PyTorch/YOLO with install buttons
                if ext == "pt" and "ultralytics" in self.model_manager.frameworks:
                    ttk.Separator(inner_frame).pack(fill=tk.X, pady=10)
                    
                    ttk.Label(
                        inner_frame,
                        text="Standard YOLOv8 Models:",
                        font=("Arial", 10, "bold")
                    ).pack(anchor=tk.W, pady=(10, 5))
                    
                    for model in self.model_manager.standard_yolo_models:
                        model_frame = ttk.Frame(inner_frame)
                        model_frame.pack(fill=tk.X, padx=5, pady=2)
                        
                        # Check if model is already available
                        installed = model in models
                        
                        if installed:
                            # Use checkbox for installed models
                            var = tk.BooleanVar(value=True)
                            cb = ttk.Checkbutton(
                                model_frame,
                                text=model,
                                variable=var
                            )
                            cb.pack(side=tk.LEFT)
                            self.model_vars[model] = var
                            
                            status_label = ttk.Label(
                                model_frame,
                                text="Installed",
                                foreground="green"
                            )
                            status_label.pack(side=tk.RIGHT)
                        else:
                            # Use label and install button for non-installed models
                            ttk.Label(
                                model_frame,
                                text=model,
                            ).pack(side=tk.LEFT)
                            
                            install_btn = ttk.Button(
                                model_frame,
                                text="Install",
                                command=lambda m=model: self._install_model(m)
                            )
                            install_btn.pack(side=tk.RIGHT)
                
                # Configure canvas scrolling
                inner_frame.update_idletasks()
                canvas.config(scrollregion=canvas.bbox("all"))
                
                # Bind mousewheel to scroll - safer implementation
                def _on_mousewheel(event, canvas=canvas):
                    try:
                        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                    except Exception as e:
                        print(f"Scrolling error: {e}")
                
                # Bind event to this specific canvas
                canvas.bind("<MouseWheel>", _on_mousewheel)
    
    def _refresh_models(self):
        """Refresh the list of available models"""
        # Scan for models again
        self.model_manager._scan_models()
        
        # Recreate the tabs
        self._create_model_tabs()
    
    def _install_model(self, model_name):
        """Install a model"""
        if self.installing:
            return
        
        self.installing = True
        
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.dialog)
        progress_dialog.title("Installing Model")
        progress_dialog.geometry("300x100")
        progress_dialog.transient(self.dialog)
        progress_dialog.grab_set()
        
        # Progress label
        progress_label = ttk.Label(
            progress_dialog,
            text=f"Installing {model_name}...\nThis may take a while.",
            wraplength=280
        )
        progress_label.pack(pady=(10, 5))
        
        # Progress bar
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_dialog,
            orient=tk.HORIZONTAL,
            mode='indeterminate',
            variable=progress_var
        )
        progress_bar.pack(fill=tk.X, padx=10, pady=5)
        progress_bar.start()
        
        # Start installation in a thread
        def install_thread():
            try:
                # Use Ultralytics to load the model (which will download it if not available)
                from ultralytics import YOLO # type: ignore
                model = YOLO(model_name)
                
                # Add to available models
                ext = model_name.split('.')[-1].lower()
                if ext not in self.model_manager.available_models:
                    self.model_manager.available_models[ext] = []
                if model_name not in self.model_manager.available_models[ext]:
                    self.model_manager.available_models[ext].append(model_name)
                
                # Store model instance
                self.model_manager.model_instances[model_name] = {"type": "yolo", "model": model}
                
                # Update UI
                progress_dialog.after(0, lambda: progress_dialog.destroy())
                self.dialog.after(0, self._refresh_models)
            except Exception as e:
                progress_dialog.after(0, lambda: progress_label.config(
                    text=f"Error installing {model_name}: {str(e)}",
                    foreground="red"
                ))
                progress_dialog.after(0, lambda: progress_bar.stop())
            finally:
                self.installing = False
        
        threading.Thread(target=install_thread, daemon=True).start()
    
    def _apply_selection(self):
        """Apply the selected models"""
        selected_models = [
            model for model, var in self.model_vars.items()
            if var.get()
        ]
        
        if self.callback:
            self.callback(selected_models)
        
        self.dialog.destroy()


# MARK: Main Application
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model Object Detection")
        self.root.geometry("1280x720")
        
        # Create model and camera managers
        self.model_manager = ModelManager()
        self.camera_manager = CameraManager()
        
        # Tracking variables
        self.running = False
        self.detection_thread = None
        self.selected_models = []
        
        # Create the UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface"""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (left side)
        control_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Camera selection
        ttk.Label(control_panel, text="Select Camera:").pack(anchor=tk.W, pady=(0, 5))
        self.camera_var = tk.StringVar()
        camera_combo = ttk.Combobox(
            control_panel, 
            textvariable=self.camera_var,
            values=list(self.camera_manager.available_cameras.keys()),
            state="readonly",
            width=20
        )
        camera_combo.pack(fill=tk.X, pady=(0, 10))
        # Select first camera by default
        if list(self.camera_manager.available_cameras.keys()):
            camera_combo.current(0)
        
        # Model selection button
        ttk.Button(
            control_panel,
            text="Install Models",
            command=self._show_model_dialog
        ).pack(fill=tk.X, pady=10)
        
        # Framework status
        framework_frame = ttk.LabelFrame(control_panel, text="Available Frameworks")
        framework_frame.pack(fill=tk.X, pady=10)
        
        for framework in ["torch", "tensorflow", "onnxruntime", "ultralytics"]:
            available = framework in self.model_manager.frameworks
            status = "Available" if available else "Not Installed"
            color = "green" if available else "red"
            
            frame_row = ttk.Frame(framework_frame)
            frame_row.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(
                frame_row, 
                text=framework.capitalize() + ":",
                width=12,
                anchor=tk.W
            ).pack(side=tk.LEFT)
            
            ttk.Label(
                frame_row,
                text=status,
                foreground=color
            ).pack(side=tk.LEFT)
        
        # Selected models display with scrollbar
        models_frame_container = ttk.Frame(control_panel)
        models_frame_container.pack(fill=tk.X, pady=10)
        
        # Selected models display with scrollbar
        ttk.Label(models_frame_container, text="Selected Models:").pack(anchor=tk.W)
        
        # Create frame with scrollbar
        models_scroll_frame = ttk.Frame(models_frame_container)
        models_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        models_scrollbar = ttk.Scrollbar(models_scroll_frame)
        models_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        models_canvas = tk.Canvas(models_scroll_frame, height=100, yscrollcommand=models_scrollbar.set)
        models_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        models_scrollbar.config(command=models_canvas.yview)
        
        # Create inner frame for models
        self.models_frame = ttk.Frame(models_canvas)
        models_canvas.create_window((0, 0), window=self.models_frame, anchor=tk.NW, width=models_canvas.winfo_width())
        
        # Configure canvas scrolling
        def configure_models_canvas(event):
            models_canvas.configure(scrollregion=models_canvas.bbox("all"))
            models_canvas.itemconfig(1, width=models_canvas.winfo_width())
        
        self.models_frame.bind("<Configure>", configure_models_canvas)
        
        # Initially show a message
        self.no_models_label = ttk.Label(
            self.models_frame,
            text="No models selected",
            foreground="gray"
        )
        self.no_models_label.pack(pady=10)
        
        # Bind mousewheel to scroll
        def _on_mousewheel(event):
            try:
                models_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except Exception as e:
                print(f"Scrolling error: {e}")
        
        models_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Start/Stop button
        self.start_stop_var = tk.StringVar(value="Start")
        self.start_stop_button = ttk.Button(
            control_panel, 
            textvariable=self.start_stop_var,
            command=self.toggle_detection,
            width=20
        )
        self.start_stop_button.pack(pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_panel, textvariable=self.status_var, wraplength=150)
        status_label.pack(pady=10)
        
        # Display panels (right side)
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Configure the grid for 2x2 layout
        for i in range(2):
            display_frame.columnconfigure(i, weight=1)
            display_frame.rowconfigure(i, weight=1)
        
        # Create four display panels
        self.panels = []
        for i in range(4):
            row = i // 2
            col = i % 2
            
            panel = DisplayPanel(
                display_frame,
                i + 1,
                self.selected_models
            )
            panel.frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.panels.append(panel)
        
        # Initially, select standard YOLO models
        self._on_models_selected(self.model_manager.standard_yolo_models)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _show_model_dialog(self):
        """Show the model selection dialog"""
        dialog = ModelSelectionDialog(self.root, self.model_manager, self._on_models_selected)
        dialog.show()
    
    def _on_models_selected(self, selected_models):
        """Handle model selection from dialog"""
        self.selected_models = selected_models
        
        # Update display panels
        for panel in self.panels:
            panel.update_models(selected_models)
        
        # Update selected models display
        for widget in self.models_frame.winfo_children():
            widget.destroy()
        
        if not selected_models:
            self.no_models_label = ttk.Label(
                self.models_frame,
                text="No models selected",
                foreground="gray"
            )
            self.no_models_label.pack(pady=10)
        else:
            for model in selected_models:
                ext = model.split('.')[-1].lower()
                
                if ext in self.model_manager.model_formats:
                    frame_name = self.model_manager.model_formats[ext]["name"]
                else:
                    frame_name = "Unknown"
                
                model_frame = ttk.Frame(self.models_frame)
                model_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ttk.Label(
                    model_frame,
                    text=model,
                    anchor=tk.W
                ).pack(side=tk.LEFT)
                
                ttk.Label(
                    model_frame,
                    text=f"({frame_name})",
                    foreground="gray"
                ).pack(side=tk.RIGHT)
            
            # Update the scrollregion after adding models
            self.models_frame.update_idletasks()
            canvas = self.models_frame.master
            canvas.configure(scrollregion=canvas.bbox("all"))
    
    def toggle_detection(self):
        """Toggle between starting and stopping detection"""
        if self.running:
            # Stop detection
            self.running = False
            self.start_stop_var.set("Start")
            self.status_var.set("Stopped")
            
            # Release camera
            self.camera_manager.release()
        else:
            # Start detection
            camera_name = self.camera_var.get()
            
            if not camera_name:
                self.status_var.set("Error: No camera selected")
                return
            
            if not self.selected_models:
                self.status_var.set("Error: No models selected")
                return
            
            # Open camera
            camera_idx = self.camera_manager.available_cameras[camera_name]
            if not self.camera_manager.open_camera(camera_idx):
                self.status_var.set(f"Error: Failed to open {camera_name}")
                return
            
            # Start detection thread
            self.running = True
            self.start_stop_var.set("Stop")
            self.status_var.set(f"Running: {camera_name}")
            
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
    
    def detection_loop(self):
        """Main detection loop running in a separate thread"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # Read frame from camera
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                self.status_var.set("Error: Failed to read frame")
                self.running = False
                break
            
            # Update each display panel
            for panel in self.panels:
                self.root.after(0, panel.update_frame, frame, self.model_manager)
            
            # Simple FPS calculation
            frame_count += 1
            if frame_count >= 10:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                self.root.after(0, self.status_var.set, f"Running: {fps:.1f} FPS")
                frame_count = 0
                start_time = time.time()
            
            # Control frame rate
            time.sleep(0.01)
    
    def on_closing(self):
        """Handle window closing event"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(1.0)  # Wait for thread to finish
        
        self.camera_manager.release()
        self.root.destroy()


# MARK: Application Entry Point
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()