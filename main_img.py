import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import os
import shutil
from PIL import Image, ImageTk
import json

from core import YOLOModelManager, DisplayPanel


class ImageManager:
    """Handles image management, uploading, and storage"""
    
    def __init__(self):
        # Create folder structure
        self.temp_folder = ".temp"
        self.original_folder = os.path.join(self.temp_folder, "original")
        self._ensure_folder_structure()
        
        # List to store uploaded images
        self.images = []
        self.current_index = 0
        
        # Scan for existing images
        self._scan_images()
    
    def _ensure_folder_structure(self):
        """Ensure all needed folders exist"""
        # Main temp folder
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        
        # Original images folder
        if not os.path.exists(self.original_folder):
            os.makedirs(self.original_folder)
    
    def _ensure_results_folder(self, model_name):
        """Ensure the results folder for a model exists and is empty"""
        result_folder = os.path.join(self.temp_folder, f"results_{model_name}")
        
        # Create folder if it doesn't exist
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        else:
            # Clear existing results
            for file in os.listdir(result_folder):
                file_path = os.path.join(result_folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        return result_folder
    
    def _scan_images(self):
        """Scan for existing images in the original folder"""
        self.images = []
        
        if os.path.exists(self.original_folder):
            for file in os.listdir(self.original_folder):
                file_path = os.path.join(self.original_folder, file)
                if os.path.isfile(file_path) and not os.path.basename(file_path).startswith('.'):
                    # Check supported image formats
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                        self.images.append(file_path)
        
        # Sort images alphabetically
        self.images.sort()
        
        # Reset current index if needed
        if not self.images:
            self.current_index = -1
        elif self.current_index >= len(self.images) or self.current_index < 0:
            self.current_index = 0
        
        return len(self.images)
    
    def get_image(self, index):
        """Get an image by index"""
        if 0 <= index < len(self.images):
            try:
                img_path = self.images[index]
                return cv2.imread(img_path), img_path
            except Exception as e:
                print(f"Error loading image at index {index}: {e}")
        
        return None, None
    
    def get_current_image(self):
        """Get the current image"""
        if not self.images or self.current_index < 0:
            return None, None
        
        return self.get_image(self.current_index)
    
    def get_image_name(self, index):
        """Get the filename of an image by index"""
        if 0 <= index < len(self.images):
            return os.path.basename(self.images[index])
        return "No image"
    
    def set_current_index(self, index):
        """Set the current image index"""
        if 0 <= index < len(self.images):
            self.current_index = index
            return True
        return False
    
    def upload_images(self):
        """Open file dialog to upload images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return 0
        
        # Copy selected files to original directory
        count = 0
        for file_path in file_paths:
            try:
                dest_path = os.path.join(self.original_folder, os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                count += 1
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
        
        # Refresh image list
        self._scan_images()
        
        return count
    
    def clear_all_images(self):
        """Delete all uploaded images and results"""
        # Delete original images
        count = 0
        if os.path.exists(self.original_folder):
            for file in os.listdir(self.original_folder):
                file_path = os.path.join(self.original_folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        
        # Clear results folders
        for folder in os.listdir(self.temp_folder):
            folder_path = os.path.join(self.temp_folder, folder)
            if os.path.isdir(folder_path) and folder.startswith("results_"):
                try:
                    shutil.rmtree(folder_path)
                    os.makedirs(folder_path)  # Recreate empty folder
                except Exception as e:
                    print(f"Error clearing results folder {folder_path}: {e}")
        
        # Refresh image list
        self._scan_images()
        
        return count
    
    def save_result(self, model_name, image_index, image, results=None, is_cls_model=False):
        """Save analysis result for an image"""
        if image_index < 0 or image_index >= len(self.images):
            return False
        
        img_path = self.images[image_index]
        img_name = os.path.basename(img_path)
        base_name, ext = os.path.splitext(img_name)
        
        # Ensure results folder exists
        result_folder = self._ensure_results_folder(model_name)
        
        if is_cls_model and results is not None:
            # For classification models, save results as JSON
            json_path = os.path.join(result_folder, f"{base_name}_result.json")
            
            # Extract classification results
            result_data = {
                "filename": img_name,
                "predictions": []
            }
            
            try:
                # Extract class predictions
                probs = results.probs
                top_indices = probs.top5
                top_probs = probs.top5conf.tolist()  # Convert tensors to list
                names = results.names
                
                # Create predictions list
                for i in range(min(5, len(top_indices))):
                    class_idx = int(top_indices[i])  # Convert tensor to int
                    prob = float(top_probs[i])  # Convert tensor to float
                    class_name = names[class_idx]
                    
                    result_data["predictions"].append({
                        "class": class_name,
                        "confidence": prob
                    })
                
                # Write JSON file
                with open(json_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                # Also save a visualization image
                result_img_path = os.path.join(result_folder, f"{base_name}_result{ext}")
                cv2.imwrite(result_img_path, image)
                
                return True
            
            except Exception as e:
                print(f"Error saving classification results: {e}")
                return False
            
        else:
            # For detection models, save the result image
            result_path = os.path.join(result_folder, f"{base_name}_result{ext}")
            
            try:
                cv2.imwrite(result_path, image)
                return True
            except Exception as e:
                print(f"Error saving result: {e}")
                return False


class ImageListFrame(ttk.Frame):
    """A scrollable frame showing image thumbnails with selection"""
    
    def __init__(self, parent, image_manager, on_select_callback):
        super().__init__(parent)
        
        self.image_manager = image_manager
        self.on_select_callback = on_select_callback
        self.buttons = []
        self.thumbnails = []  # Keep references to prevent garbage collection
        
        # Create scrollable canvas
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Inner frame for content
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Add selection highlight style
        self.style = ttk.Style()
        self.style.configure("Selected.TButton", background="#4a7abc")
        
        # Initialize empty list
        self.refresh_list()
    
    def _on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def refresh_list(self):
        """Refresh the image list"""
        # Clear existing buttons
        for button in self.buttons:
            button.destroy()
        
        self.buttons = []
        self.thumbnails = []
        
        # No images
        if not self.image_manager.images:
            empty_label = ttk.Label(self.inner_frame, text="No images available")
            empty_label.pack(pady=10)
            return
        
        # Create a button for each image
        for i, img_path in enumerate(self.image_manager.images):
            # Create frame for this item
            item_frame = ttk.Frame(self.inner_frame)
            item_frame.pack(fill="x", padx=5, pady=2)
            
            # Create thumbnail
            try:
                # Load and resize thumbnail
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (60, 60))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    thumb = Image.fromarray(img)
                    thumb_tk = ImageTk.PhotoImage(image=thumb)
                    self.thumbnails.append(thumb_tk)  # Keep reference
                    
                    # Create thumbnail label
                    thumb_label = ttk.Label(item_frame, image=thumb_tk)
                    thumb_label.pack(side="left", padx=(0, 5))
                else:
                    # Fallback for unreadable images
                    thumb_label = ttk.Label(item_frame, text="[No Preview]", width=10)
                    thumb_label.pack(side="left", padx=(0, 5))
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
                thumb_label = ttk.Label(item_frame, text="[Error]", width=10)
                thumb_label.pack(side="left", padx=(0, 5))
            
            # Create button with image name
            img_name = os.path.basename(img_path)
            button = ttk.Button(
                item_frame,
                text=img_name,
                command=lambda idx=i: self.select_image(idx),
                style="TButton" if i != self.image_manager.current_index else "Selected.TButton"
            )
            button.pack(side="left", fill="x", expand=True)
            
            self.buttons.append(button)
        
        # Update selection
        self.update_selection()
    
    def select_image(self, index):
        """Select an image by index"""
        if self.image_manager.set_current_index(index):
            self.update_selection()
            if self.on_select_callback:
                self.on_select_callback(index)
    
    def update_selection(self):
        """Update button styles to show selection"""
        current = self.image_manager.current_index
        for i, button in enumerate(self.buttons):
            if i == current:
                button.configure(style="Selected.TButton")
            else:
                button.configure(style="TButton")


# Main Application
class YOLOImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Analysis")
        self.root.geometry("1280x720")
        
        # Create model and image managers
        self.model_manager = YOLOModelManager()
        self.image_manager = ImageManager()
        
        # Results dictionary to store pre-computed results for faster display
        # Structure: {image_index: {model_name: {'image': processed_img, 'results': results, 'is_cls_model': bool}}}
        self.results_cache = {}
        
        # Tracking variables
        self.analyzing = False
        self.analysis_thread = None
        self.selected_models = []
        self.model_checkboxes = {}
        self.installing = False
        
        # Create the UI
        self._create_ui()
        
        # Update the UI with the current image
        self.update_image_display()
    
    def _create_ui(self):
        """Create the user interface"""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (left side)
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), expand=False)
        
        # Top controls section
        top_controls = ttk.LabelFrame(control_panel, text="Controls", padding="10")
        top_controls.pack(fill=tk.X, pady=(0, 10))
        
        # Upload and clear buttons
        button_frame = ttk.Frame(top_controls)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="Upload Images",
            command=self.upload_images
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        ttk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_images
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Ultralytics check
        if not self.model_manager.ultralytics_available:
            ttk.Label(
                top_controls,
                text="Ultralytics is required for YOLO models.\nInstall with: pip install ultralytics",
                foreground="red",
                justify=tk.CENTER
            ).pack(fill=tk.X, pady=5)
        
        # Image list section (scrollable)
        images_frame = ttk.LabelFrame(control_panel, text="Images")
        images_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_list = ImageListFrame(
            images_frame,
            self.image_manager,
            self.on_image_selected
        )
        self.image_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Models selection section
        models_frame = ttk.LabelFrame(control_panel, text="Models")
        models_frame.pack(fill=tk.X, pady=(0, 10))
        
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
        
        # Analysis buttons section
        analysis_frame = ttk.LabelFrame(control_panel, text="Analysis")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Analysis buttons
        self.analyze_button = ttk.Button(
            analysis_frame, 
            text="Analyze Selected Image",
            command=self.analyze_current_image
        )
        self.analyze_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.analyze_all_button = ttk.Button(
            analysis_frame, 
            text="Analyze All Images",
            command=self.analyze_all_images
        )
        self.analyze_all_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(control_panel, textvariable=self.status_var, wraplength=200)
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
    
    def upload_images(self):
        """Upload images via file dialog"""
        count = self.image_manager.upload_images()
        if count > 0:
            self.status_var.set(f"Uploaded {count} images")
            # Refresh the image list UI
            self.image_list.refresh_list()
            # Update display
            self.update_image_display()
        else:
            self.status_var.set("No images selected")
    
    def clear_images(self):
        """Clear all uploaded images"""
        if not self.image_manager.images:
            messagebox.showinfo("Info", "No images to clear")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all images?"):
            count = self.image_manager.clear_all_images()
            self.status_var.set(f"Cleared {count} images")
            self.results_cache = {}  # Clear the results cache
            
            # Refresh the image list UI
            self.image_list.refresh_list()
            # Update display
            self.update_image_display()
    
    def on_image_selected(self, index):
        """Handler for image selection"""
        self.update_image_display()
    
    def update_image_display(self):
        """Update the UI to display the current image"""
        # Get current image
        frame, _ = self.image_manager.get_current_image()
        
        if frame is None:
            # Clear all displays
            for panel in self.panels:
                self.root.after(0, lambda p=panel: p.update_frame(None))
            return
        
        # Get the current index
        current_idx = self.image_manager.current_index
        
        # Update result displays based on cached results
        for panel in self.panels:
            display_type = panel.display_var.get()
            
            if display_type == "None":
                # Skip, panel will handle this case
                pass
            elif display_type == "Original":
                self.root.after(0, lambda p=panel, f=frame: p.update_frame(f))
            else:
                # Check if we have cached results
                model_name = display_type
                if (current_idx in self.results_cache and 
                    model_name in self.results_cache[current_idx]):
                    # Use cached results
                    result_data = self.results_cache[current_idx][model_name]
                    self.root.after(0, lambda p=panel, f=frame, m=model_name, r=result_data: 
                                   p.update_frame(f, results_dict={m: r}))
                else:
                    # No cached results, just show original frame
                    self.root.after(0, lambda p=panel, f=frame: p.update_frame(f))
    
    def analyze_current_image(self):
        """Analyze the current image with selected models"""
        if self.analyzing:
            messagebox.showinfo("Info", "Analysis is already in progress")
            return
        
        if not self.image_manager.images:
            messagebox.showinfo("Info", "No images to analyze")
            return
        
        if not self.selected_models:
            messagebox.showinfo("Info", "No models selected")
            return
        
        if not self.model_manager.ultralytics_available:
            messagebox.showinfo("Error", "Ultralytics not installed")
            return
        
        # Start analysis thread
        self.analyzing = True
        self.status_var.set("Analyzing...")
        self.analyze_button.config(state=tk.DISABLED)
        self.analyze_all_button.config(state=tk.DISABLED)
        
        current_idx = self.image_manager.current_index
        
        def analysis_thread():
            try:
                frame, _ = self.image_manager.get_current_image()
                if frame is None:
                    self.root.after(0, lambda: self.status_var.set("Error: Failed to load image"))
                    return
                
                # Initialize cache for this image if needed
                if current_idx not in self.results_cache:
                    self.results_cache[current_idx] = {}
                
                # Process with each selected model
                for model_name in self.selected_models:
                    try:
                        # Update status
                        self.root.after(0, lambda m=model_name: self.status_var.set(f"Analyzing with {m}..."))
                        
                        # Run inference
                        annotated_frame, results, is_cls_model = self.model_manager.predict(frame, model_name)
                        
                        # Cache the results
                        self.results_cache[current_idx][model_name] = {
                            'image': annotated_frame,
                            'results': results,
                            'is_cls_model': is_cls_model
                        }
                        
                        # Save results to disk
                        self.image_manager.save_result(model_name, current_idx, annotated_frame, results, is_cls_model)
                        
                        # Update display panels
                        for panel in self.panels:
                            if panel.display_var.get() == model_name:
                                result_data = {model_name: self.results_cache[current_idx][model_name]}
                                self.root.after(0, lambda p=panel, f=frame, rd=result_data: 
                                               p.update_frame(f, results_dict=rd))
                    
                    except Exception as e:
                        self.root.after(0, lambda err=e: self.status_var.set(f"Error with {model_name}: {str(err)}"))
                        print(f"Error with {model_name}: {e}")
                
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
            
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            
            finally:
                self.analyzing = False
                self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.analyze_all_button.config(state=tk.NORMAL))


# Application Entry Point
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOImageApp(root)
    root.mainloop()