import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
import os

from core import YOLOModelManager, DisplayPanel


# Camera Manager
class CameraManager:
    """Handles camera detection and streaming"""
    
    def __init__(self):
        self.available_cameras = self._detect_cameras()
        self.current_camera = None
        self.current_camera_index = None
    
    def _detect_cameras(self):
        """Detect available cameras on the system"""
        available_cameras = {}
        # Try camera indices (0 to 10)
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


# Main Application
class YOLOCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Camera Detection")
        self.root.geometry("1280x720")
        
        # Create model and camera managers
        self.model_manager = YOLOModelManager()
        self.camera_manager = CameraManager()
        
        # Tracking variables
        self.running = False
        self.detection_thread = None
        self.selected_models = []
        self.model_checkboxes = {}
        self.installing = False
        
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
        
        # Ultralytics check
        if not self.model_manager.ultralytics_available:
            ttk.Label(
                control_panel,
                text="Ultralytics is required for YOLO models.\nInstall with: pip install ultralytics",
                foreground="red",
                justify=tk.CENTER
            ).pack(fill=tk.X, pady=5)
        
        # Models selection frame
        models_frame = ttk.LabelFrame(control_panel, text="Select Models")
        models_frame.pack(fill=tk.X, pady=10)
        
        # Create inner frame with scrollbar for models
        models_scroll_frame = ttk.Frame(models_frame)
        models_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        models_scrollbar = ttk.Scrollbar(models_scroll_frame)
        models_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        models_canvas = tk.Canvas(models_scroll_frame, height=150, yscrollcommand=models_scrollbar.set)
        models_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        models_scrollbar.config(command=models_canvas.yview)
        
        self.models_inner_frame = ttk.Frame(models_canvas)
        models_canvas.create_window((0, 0), window=self.models_inner_frame, anchor=tk.NW, width=models_canvas.winfo_width())
        
        # Configure canvas scrolling
        def configure_models_canvas(event):
            models_canvas.configure(scrollregion=models_canvas.bbox("all"))
            models_canvas.itemconfig(1, width=models_canvas.winfo_width())
        
        self.models_inner_frame.bind("<Configure>", configure_models_canvas)
        
        # Bind mousewheel to scroll
        def _on_mousewheel(event):
            try:
                models_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except Exception as e:
                print(f"Scrolling error: {e}")
        
        models_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Refresh models button
        ttk.Button(
            models_frame,
            text="Refresh Models",
            command=self._refresh_models
        ).pack(fill=tk.X, padx=5, pady=5)
        
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
                []  # Empty initially, will update after model selection
            )
            panel.frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.panels.append(panel)
        
        # Now populate models after panels are created
        self._populate_models()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _populate_models(self):
        """Populate the models list with checkboxes and install buttons"""
        # Clear existing widgets
        for widget in self.models_inner_frame.winfo_children():
            widget.destroy()
        
        # Reset checkboxes dictionary
        self.model_checkboxes = {}
        
        # Get available models
        available_models = self.model_manager.available_models
        
        # Add installed models with checkboxes
        for model in available_models:
            model_frame = ttk.Frame(self.models_inner_frame)
            model_frame.pack(fill=tk.X, pady=2)
            
            var = tk.BooleanVar(value=True)  # Selected by default
            cb = ttk.Checkbutton(
                model_frame,
                text=model,
                variable=var,
                command=self._update_selected_models
            )
            cb.pack(side=tk.LEFT)
            
            # Add to checkbox dict
            self.model_checkboxes[model] = var
        
        # Add standard models that aren't already installed
        for model in self.model_manager.standard_yolo_models:
            if model not in available_models:
                model_frame = ttk.Frame(self.models_inner_frame)
                model_frame.pack(fill=tk.X, pady=2)
                
                ttk.Label(
                    model_frame,
                    text=model
                ).pack(side=tk.LEFT)
                
                install_btn = ttk.Button(
                    model_frame,
                    text="Install",
                    command=lambda m=model: self._install_model(m)
                )
                install_btn.pack(side=tk.RIGHT)
        
        # Update selected models list
        self._update_selected_models()
    
    def _refresh_models(self):
        """Refresh the list of available models"""
        # Scan for models again
        self.model_manager.available_models = self.model_manager._scan_models()
        
        # Repopulate the models list
        self._populate_models()
    
    def _update_selected_models(self):
        """Update the list of selected models based on checkboxes"""
        self.selected_models = [
            model for model, var in self.model_checkboxes.items()
            if var.get()
        ]
        
        # Update display panels with selected models
        for panel in self.panels:
            panel.update_models(self.selected_models)
    
    def _install_model(self, model_name):
        """Install a YOLO model"""
        if self.installing or not self.model_manager.ultralytics_available:
            return
        
        self.installing = True
        
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Installing Model")
        progress_dialog.geometry("300x100")
        progress_dialog.transient(self.root)
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
                from ultralytics import YOLO
                model = YOLO(model_name)
                
                # Add to available models
                if model_name not in self.model_manager.available_models:
                    self.model_manager.available_models.append(model_name)
                
                # Store model instance
                self.model_manager.model_instances[model_name] = model
                
                # Update UI
                progress_dialog.after(0, lambda: progress_dialog.destroy())
                self.root.after(0, self._refresh_models)
            except Exception as e:
                progress_dialog.after(0, lambda: progress_label.config(
                    text=f"Error installing {model_name}: {str(e)}",
                    foreground="red"
                ))
                progress_dialog.after(0, lambda: progress_bar.stop())
            finally:
                self.installing = False
        
        threading.Thread(target=install_thread, daemon=True).start()
    
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
            
            if not self.model_manager.ultralytics_available:
                self.status_var.set("Error: Ultralytics not installed")
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
                self.root.after(0, lambda p=panel, f=frame: p.update_frame(f, self.model_manager))
            
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


# Application Entry Point
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOCameraApp(root)
    root.mainloop()