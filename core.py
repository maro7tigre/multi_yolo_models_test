import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import threading
import time
import os
from PIL import Image, ImageTk

# YOLO Model Manager for both camera and image-based applications
class YOLOModelManager:
    """Handles loading and inference for YOLO models"""
    
    def __init__(self):
        # Check for Ultralytics YOLO
        self.ultralytics_available = self._check_ultralytics()
        print(f"Ultralytics available: {self.ultralytics_available}")
        
        # Scan for available YOLO models
        self.available_models = self._scan_models()
        
        # Store loaded model instances
        self.model_instances = {}
        
        # Standard YOLOv8 models
        self.standard_yolo_models = [
            "yolov8n-cls.pt", 
            "yolov8s-cls.pt", 
            "yolov8m-cls.pt", 
            "yolov8l-cls.pt", 
            "yolov8x-cls.pt"
        ]
    
    def _check_ultralytics(self):
        """Check if Ultralytics YOLO is available"""
        try:
            import ultralytics
            return True
        except ImportError:
            return False
    
    def _scan_models(self):
        """Scan for available .pt model files"""
        models = []
        for file in os.listdir('.'):
            if file.lower().endswith('.pt'):
                models.append(file)
        return models
    
    def load_model(self, model_name):
        """Load a YOLO model"""
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        
        if not self.ultralytics_available:
            raise ValueError("Ultralytics YOLO is not installed")
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)
            self.model_instances[model_name] = model
            return model
        except Exception as e:
            print(f"Error loading YOLO model {model_name}: {e}")
            return None
    
    def predict(self, frame, model_name):
        """Run inference on a frame using the specified YOLO model"""
        if model_name not in self.model_instances:
            model = self.load_model(model_name)
            if model is None:
                return frame, None, False
        else:
            model = self.model_instances[model_name]
        
        try:
            # Check if this is a classification model based on the model name
            is_cls_model = "-cls" in model_name.lower()
            
            results = model(frame)
            
            if is_cls_model:
                # For classification models, we'll return the original frame and results
                # The third parameter indicates this is a classification model
                return frame, results[0], True
            else:
                # For detection models, return the annotated frame as before
                annotated_frame = results[0].plot()
                return annotated_frame, results[0], False
        except Exception as e:
            print(f"Error during inference with {model_name}: {e}")
            # Return original frame if inference fails
            return frame, None, False


# Display Panel for both camera and image applications
class DisplayPanel:
    """A panel that can display either raw image/camera feed or processed model output"""
    
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
        
        # Create options: "None", "Original", and all models
        options = ["None", "Original"] + models
        
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
        options = ["None", "Original"] + models
        self.display_combo['values'] = options
    
    def update_frame(self, frame, model_manager=None, results_dict=None):
        """
        Update the display with a new frame
        
        Args:
            frame: The original image frame
            model_manager: YOLO model manager (for live inference)
            results_dict: Dictionary of pre-computed results (for image app)
        """
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
        
        elif display_type == "Original":
            # Display raw input
            display_frame = frame
            self._display_image(display_frame)
        
        else:
            # Display model processed frame
            model_name = display_type
            try:
                if results_dict and model_name in results_dict:
                    # Use pre-computed results (for image app)
                    result_data = results_dict[model_name]
                    display_frame = result_data.get('image')
                    results = result_data.get('results')
                    is_cls_model = result_data.get('is_cls_model', False)
                    
                    if is_cls_model and results is not None:
                        self._display_cls_results(frame, results)
                    else:
                        self._display_image(display_frame)
                        
                elif model_manager:
                    # Process with the selected model (for camera app)
                    display_frame, results, is_cls_model = model_manager.predict(frame, model_name)
                    
                    if is_cls_model and results is not None:
                        # For classification models, display text results instead of image
                        self._display_cls_results(frame, results)
                    else:
                        # For detection models, display the annotated image
                        self._display_image(display_frame)
                else:
                    # No results available
                    self.canvas.delete("all")
                    self.canvas.create_text(
                        self.canvas.winfo_width() // 2,
                        self.canvas.winfo_height() // 2,
                        text="No results available",
                        fill="yellow",
                        font=('Arial', 12)
                    )
                                
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

    def _display_image(self, frame):
        """Display an image frame on the canvas"""
        # Convert frame to Tkinter PhotoImage
        self.photo_img = self._convert_frame_to_tk(frame)
        
        if self.photo_img:
            # Display the image
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self.photo_img
            )

    def _display_cls_results(self, frame, results):
        """Display classification results as text"""
        self.canvas.delete("all")
        
        # Draw a dark background
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_rectangle(0, 0, w, h, fill="black")
        
        # Start y position for text
        y_pos = 20
        
        # Display title
        self.canvas.create_text(
            w // 2, 
            y_pos,
            text="Classification Results",
            fill="white",
            font=('Arial', 16, 'bold')
        )
        y_pos += 40
        
        # Get top 5 predictions from the results
        try:
            # Extract class predictions from the results
            probs = results.probs
            
            # Get top 5 class indices and their probabilities
            top_indices = probs.top5
            top_probs = probs.top5conf
            
            # Get class names
            names = results.names
            
            # Display each prediction
            for i in range(min(5, len(top_indices))):
                class_idx = top_indices[i]
                prob = top_probs[i] * 100  # Convert to percentage
                class_name = names[class_idx]
                
                # Create colored confidence indicator
                if prob > 75:
                    color = "#4CAF50"  # Green for high confidence
                elif prob > 50:
                    color = "#FFC107"  # Amber for medium confidence
                else:
                    color = "#F44336"  # Red for low confidence
                
                # Draw confidence bar
                bar_width = int((w - 60) * (prob / 100))
                self.canvas.create_rectangle(
                    30, y_pos+5, 30 + bar_width, y_pos+25, 
                    fill=color, outline=""
                )
                
                # Display class name and probability
                self.canvas.create_text(
                    w // 2, 
                    y_pos + 15,
                    text=f"{class_name}: {prob:.1f}%",
                    fill="white",
                    font=('Arial', 12)
                )
                
                y_pos += 30
            
        except Exception as e:
            self.canvas.create_text(
                w // 2, 
                h // 2,
                text=f"Error processing results: {str(e)}",
                fill="red",
                font=('Arial', 12),
                width=w - 40
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