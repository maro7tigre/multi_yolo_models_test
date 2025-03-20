import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
import os
from PIL import Image, ImageTk
from ultralytics import YOLO

# MARK: Model Manager
class ModelManager:
    """Handles the loading and management of YOLO models"""
    
    def __init__(self):
        # Standard YOLOv8 models
        self.standard_models = [
            "yolov8n.pt", 
            "yolov8s.pt", 
            "yolov8m.pt", 
            "yolov8l.pt", 
            "yolov8x.pt"
        ]
        # Check which models are available locally
        self.available_models = self._get_available_models()
        # Dict to store loaded model instances
        self.model_instances = {}
    
    def _get_available_models(self):
        """Check which models are available locally"""
        available = []
        for file in os.listdir('.'):
            if file.endswith('.pt'):
                available.append(file)
        return available
    
    def load_model(self, model_name):
        """Load a YOLO model by name (will download if not available)"""
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        
        try:
            # YOLO will download the model if it's not available locally
            model = YOLO(model_name)
            self.model_instances[model_name] = model
            # Add to available models if not already there
            if model_name not in self.available_models:
                self.available_models.append(model_name)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, frame, model_name):
        """Run inference on a frame using specified model"""
        if model_name not in self.model_instances:
            model = self.load_model(model_name)
            if model is None:
                return frame, None
        else:
            model = self.model_instances[model_name]
        
        results = model(frame)
        annotated_frame = results[0].plot()
        
        return annotated_frame, results[0]


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


# MARK: Main Application
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.geometry("1280x720")
        
        # Create model and camera managers
        self.model_manager = ModelManager()
        self.camera_manager = CameraManager()
        
        # Tracking variables
        self.running = False
        self.detection_thread = None
        
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
        
        # Model selection
        model_frame = ttk.LabelFrame(control_panel, text="Available Models")
        model_frame.pack(fill=tk.X, pady=10)
        
        # Checkboxes for each model
        self.model_vars = {}
        for model in self.model_manager.standard_models:
            var = tk.BooleanVar(value=model in self.model_manager.available_models)
            
            cb = ttk.Checkbutton(
                model_frame,
                text=model,
                variable=var,
                command=self._update_models
            )
            cb.pack(anchor=tk.W, padx=5, pady=2)
            self.model_vars[model] = var
        
        # Info label
        ttk.Label(
            model_frame,
            text="Note: Models will be downloaded\nautomatically when needed.",
            foreground="gray",
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=5, pady=5)
        
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
                self.get_selected_models()
            )
            panel.frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.panels.append(panel)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _update_models(self):
        """Update the list of selected models"""
        # Get the list of selected models
        models = self.get_selected_models()
        
        # Update each display panel
        for panel in self.panels:
            panel.update_models(models)
    
    def get_selected_models(self):
        """Get list of models that are selected"""
        return [model for model, var in self.model_vars.items() if var.get()]
    
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