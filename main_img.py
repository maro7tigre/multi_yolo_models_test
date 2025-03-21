import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import threading
import time
import shutil
import random
import glob
from PIL import Image, ImageTk

# Import the YOLOModelManager and DisplayPanel from core.py
from core import YOLOModelManager, DisplayPanel

class YOLOImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Analyzer")
        self.root.geometry("1280x720")
        
        # Create model manager
        self.model_manager = YOLOModelManager()
        
        # Tracking variables
        self.images = []  # List of {path, name, data}
        self.current_image_index = -1
        self.selected_models = []
        self.model_checkboxes = {}
        self.analyzing = False
        self.results_dict = {}  # Store results for all models
        
        # Set up temp directory for random images
        self.temp_dir = os.path.join('.', '.temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
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
        
        # File section
        file_frame = ttk.LabelFrame(control_panel, text="Images")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            file_frame,
            text="Open Images",
            command=self.open_images
        ).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            file_frame,
            text="Get Random Images",
            command=self.get_random_images
        ).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            file_frame,
            text="Clear Images",
            command=self.clear_images
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Image listbox
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.file_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            exportselection=0,
            height=10
        )
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.file_listbox.config(yscrollcommand=list_scrollbar.set)
        list_scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
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
        """Populate the models list with checkboxes"""
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
        
        # If an image is already selected, re-analyze it with the new model selection
        if self.current_image_index >= 0 and self.selected_models:
            self.analyze_current_image()
    
    def open_images(self):
        """Open image files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        # Clear existing images if there are any
        if self.images:
            if not messagebox.askyesno("Clear Images", 
                                     "This will clear your existing images. Continue?"):
                return
            self.clear_images(ask=False)
        
        # Load selected images
        count = 0
        for path in file_paths:
            try:
                # Read image
                img_data = cv2.imread(path)
                if img_data is None:
                    continue
                
                # Get file name
                name = os.path.basename(path)
                
                # Add to our list
                self.images.append({
                    'path': path,
                    'name': name,
                    'data': img_data
                })
                
                # Add to listbox
                self.file_listbox.insert(tk.END, name)
                
                count += 1
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Select first image if available
        if count > 0:
            self.file_listbox.selection_set(0)
            self.current_image_index = 0
            self.analyze_current_image()
        
        # Update status
        self.status_var.set(f"Loaded {count} images")
    
    def clear_images(self, ask=True):
        """Clear all loaded images"""
        if ask and self.images:
            if not messagebox.askyesno("Clear Images", "Really clear all images?"):
                return
        
        # Clear image data
        self.images = []
        self.current_image_index = -1
        self.results_dict = {}
        
        # Clear UI
        self.file_listbox.delete(0, tk.END)
        
        # Clear all panels
        for panel in self.panels:
            panel.update_frame(None)
        
        # Update status
        self.status_var.set("All images cleared")
    
    def on_file_select(self, event):
        """Handle image selection from listbox"""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if 0 <= index < len(self.images):
            self.current_image_index = index
            self.analyze_current_image()
    
    def analyze_current_image(self):
        """Analyze the current image with all selected models"""
        if self.analyzing or self.current_image_index < 0:
            return
        
        if not self.selected_models:
            # Just display the original image
            self.update_panels_with_original()
            return
        
        # Start analysis
        self.analyzing = True
        self.status_var.set("Analyzing image...")
        
        # Get the image to analyze
        img_data = self.images[self.current_image_index]
        image = img_data['data'].copy()
        
        # Clear results dictionary for this image
        self.results_dict = {}
        
        # Show original image in first panel
        self.panels[0].display_var.set("Original")
        self.panels[0].update_frame(image)
        
        # Run analysis in a thread
        def analysis_thread():
            try:
                # Process each selected model
                for model_name in self.selected_models:
                    if not self.analyzing:  # Check if analysis was cancelled
                        break
                    
                    self.root.after(0, lambda m=model_name: 
                        self.status_var.set(f"Analyzing with {m}..."))
                    
                    # Load model if needed
                    if model_name not in self.model_manager.model_instances:
                        self.model_manager.load_model(model_name)
                    
                    # Run inference
                    frame_copy = image.copy()
                    annotated_frame, results, is_cls_model = self.model_manager.predict(
                        frame_copy, model_name
                    )
                    
                    # Store results
                    self.results_dict[model_name] = {
                        'image': annotated_frame,
                        'results': results,
                        'is_cls_model': is_cls_model
                    }
                
                # Update all panels with latest results
                self.root.after(0, self.update_panels_with_results)
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
                
            except Exception as e:
                err_msg = str(e)
                print(f"Analysis error: {e}")
                self.root.after(0, lambda err=err_msg: 
                    self.status_var.set(f"Error: {err}"))
            
            finally:
                self.analyzing = False
        
        # Start the thread
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def update_panels_with_original(self):
        """Update panels with just the original image"""
        if self.current_image_index < 0:
            return
            
        # Get the original image
        image = self.images[self.current_image_index]['data']
        
        # Show original image in first panel
        self.panels[0].display_var.set("Original")
        self.panels[0].update_frame(image)
        
        # Clear other panels
        for i in range(1, 4):
            self.panels[i].display_var.set("None")
            self.panels[i].update_frame(None)
    
    def update_panels_with_results(self):
        """Update all panels with the latest results"""
        if self.current_image_index < 0:
            return
        
        # Get the original image
        image = self.images[self.current_image_index]['data']
        
        # Update first panel with original image
        self.panels[0].display_var.set("Original")
        self.panels[0].update_frame(image)
        
        # Update other panels with model results if available
        for i in range(1, 4):
            panel = self.panels[i]
            
            # Get selected display type
            display_type = panel.display_var.get()
            
            if display_type == "Original":
                panel.update_frame(image)
            elif display_type != "None" and display_type in self.results_dict:
                # Show model results
                panel.update_frame(image, None, self.results_dict)
    
    def get_random_images(self):
        """Get random images from dataset/val subdirectories"""
        # Check if dataset/val exists
        dataset_path = os.path.join('.', 'dataset', 'val')
        if not os.path.isdir(dataset_path):
            messagebox.showwarning(
                "Dataset Not Found", 
                "The 'dataset/val' directory was not found. Please create it and add class subdirectories with images."
            )
            return
        
        # Clear temp directory
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            else:
                os.makedirs(self.temp_dir)
                
            self.status_var.set("Cleared temp directory")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear temp directory: {str(e)}")
            return
        
        # Get subdirectories (classes)
        class_dirs = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                class_dirs.append(item_path)
        
        if not class_dirs:
            messagebox.showwarning(
                "No Class Directories", 
                "No class subdirectories found in 'dataset/val'. Please create class subdirectories with images."
            )
            return
        
        # Get random images from each class
        all_images = []
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            # Get all image files
            image_files = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                image_files.extend(glob.glob(os.path.join(class_dir, ext)))
            
            # Select up to 10 random images from this class
            if image_files:
                samples = min(10, len(image_files))
                selected_images = random.sample(image_files, samples)
                all_images.extend([(img, class_name) for img in selected_images])
        
        if not all_images:
            messagebox.showwarning(
                "No Images Found", 
                "No images found in the class subdirectories. Please add images to the class directories."
            )
            return
        
        # Shuffle all selected images
        random.shuffle(all_images)
        
        # Copy to temp directory with numbered prefixes
        temp_image_paths = []
        for i, (img_path, class_name) in enumerate(all_images):
            # Get original filename and construct new filename
            orig_name = os.path.basename(img_path)
            extension = os.path.splitext(orig_name)[1]
            new_name = f"{i+1:03d}_{class_name}{extension}"
            dest_path = os.path.join(self.temp_dir, new_name)
            
            # Copy the file
            try:
                shutil.copy2(img_path, dest_path)
                temp_image_paths.append(dest_path)
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
        
        # Clear current images and load the temp images
        self.clear_images(ask=False)
        
        # Open the temp images
        if temp_image_paths:
            count = 0
            for path in temp_image_paths:
                try:
                    # Read image
                    img_data = cv2.imread(path)
                    if img_data is None:
                        continue
                    
                    # Get file name
                    name = os.path.basename(path)
                    
                    # Add to our list
                    self.images.append({
                        'path': path,
                        'name': name,
                        'data': img_data
                    })
                    
                    # Add to listbox
                    self.file_listbox.insert(tk.END, name)
                    
                    count += 1
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            # Select first image if available
            if count > 0:
                self.file_listbox.selection_set(0)
                self.current_image_index = 0
                self.analyze_current_image()
            
            # Update status
            self.status_var.set(f"Loaded {count} random images")
        else:
            self.status_var.set("No images were loaded")
    
    def on_closing(self):
        """Handle window closing event"""
        self.analyzing = False
        self.root.destroy()


# Application Entry Point
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOImageAnalyzerApp(root)
    root.mainloop()