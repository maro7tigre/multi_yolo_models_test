import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import json
import threading
from PIL import Image, ImageTk

# Import the YOLOModelManager from core.py
from core import YOLOModelManager

class YOLOImageAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Analyzer")
        self.root.geometry("1200x700")
        
        # Initialize model manager
        self.model_manager = YOLOModelManager()
        
        # Application state
        self.images = []  # List of {path, name, image_data}
        self.current_image_index = -1
        self.selected_model = None
        self.analyzing = False
        
        # Set up results directory
        self.results_dir = os.path.join('.', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create UI elements
        self.create_ui()
        
        # Check for YOLO models
        self.check_models()
    
    def create_ui(self):
        """Create the main UI layout"""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Split into left sidebar and right content area
        left_frame = ttk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)  # Maintain width
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create sidebar components
        self.create_sidebar(left_frame)
        
        # Create main content components
        self.create_main_content(right_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            padding=(5, 2),
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_label.pack(fill=tk.X)
    
    def create_sidebar(self, parent):
        """Create the sidebar with file list and controls"""
        # File controls section
        file_frame = ttk.LabelFrame(parent, text="Images")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_buttons = ttk.Frame(file_frame)
        file_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            file_buttons, 
            text="Open Images",
            command=self.open_images
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        ttk.Button(
            file_buttons, 
            text="Clear All",
            command=self.clear_images
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))
        
        # File list
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.file_listbox = tk.Listbox(
            list_frame, 
            selectmode=tk.SINGLE,
            exportselection=0
        )
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.file_listbox.config(yscrollcommand=list_scrollbar.set)
        list_scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # Model selection section
        model_frame = ttk.LabelFrame(parent, text="Models")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_listbox = tk.Listbox(
            model_frame, 
            selectmode=tk.SINGLE,
            exportselection=0,
            height=6
        )
        
        model_scrollbar = ttk.Scrollbar(model_frame, orient=tk.VERTICAL)
        self.model_listbox.config(yscrollcommand=model_scrollbar.set)
        model_scrollbar.config(command=self.model_listbox.yview)
        
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # Bind selection event
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        
        # Refresh models button
        ttk.Button(
            model_frame, 
            text="Refresh Models",
            command=self.refresh_models
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Analysis controls section
        analysis_frame = ttk.LabelFrame(parent, text="Analysis")
        analysis_frame.pack(fill=tk.X)
        
        self.analyze_button = ttk.Button(
            analysis_frame, 
            text="Analyze Image",
            command=self.analyze_current,
            state=tk.DISABLED
        )
        self.analyze_button.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            analysis_frame, 
            text="Export Results",
            command=self.export_results
        ).pack(fill=tk.X, padx=5, pady=5)
    
    def create_main_content(self, parent):
        """Create the main content area with image and results display"""
        # Top section: Image display
        image_frame = ttk.Frame(parent)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Split into two columns for original and processed images
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Original image
        original_frame = ttk.LabelFrame(image_frame, text="Original Image")
        original_frame.grid(row=0, column=0, padx=(0, 5), pady=0, sticky='nsew')
        
        self.original_canvas = tk.Canvas(original_frame, bg='black')
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Processed image
        processed_frame = ttk.LabelFrame(image_frame, text="Processed Image")
        processed_frame.grid(row=0, column=1, padx=(5, 0), pady=0, sticky='nsew')
        
        self.processed_canvas = tk.Canvas(processed_frame, bg='black')
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom section: Results
        results_frame = ttk.LabelFrame(parent, text="Detection Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results tree view
        columns = ("Class", "Confidence", "Coordinates")
        self.results_tree = ttk.Treeview(
            results_frame, 
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        
        # Configure columns
        self.results_tree.heading("Class", text="Class")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Coordinates", text="Coordinates")
        
        self.results_tree.column("Class", width=150)
        self.results_tree.column("Confidence", width=100)
        self.results_tree.column("Coordinates", width=300)
        
        # Add scrollbar
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL)
        self.results_tree.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.results_tree.yview)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # Reference to displayed images to prevent garbage collection
        self.displayed_original = None
        self.displayed_processed = None
    
    def check_models(self):
        """Check for available YOLO models"""
        if not self.model_manager.ultralytics_available:
            self.status_var.set("Ultralytics not installed. Install with: pip install ultralytics")
            messagebox.showwarning(
                "Ultralytics Not Found", 
                "The Ultralytics package is required to use YOLO models.\n"
                "Install it with: pip install ultralytics"
            )
        else:
            self.status_var.set(f"Ultralytics available: {self.model_manager.ultralytics_available}")
            self.refresh_models()
    
    def refresh_models(self):
        """Refresh the list of available models"""
        # Scan for models again
        self.model_manager.available_models = self.model_manager._scan_models()
        
        # Update the listbox
        self.model_listbox.delete(0, tk.END)
        
        for model in self.model_manager.available_models:
            self.model_listbox.insert(tk.END, model)
        
        # Update UI state
        self.update_ui_state()
    
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
            self.display_image(self.images[0])
            
            # Try to load existing results
            self.check_for_results()
        
        # Update status
        self.status_var.set(f"Loaded {count} images")
        
        # Update UI state
        self.update_ui_state()
    
    def clear_images(self, ask=True):
        """Clear all loaded images"""
        if ask and self.images:
            if not messagebox.askyesno("Clear Images", "Really clear all images?"):
                return
        
        # Clear image data
        self.images = []
        self.current_image_index = -1
        
        # Clear UI
        self.file_listbox.delete(0, tk.END)
        self.clear_display()
        self.clear_results()
        
        # Update UI state
        self.update_ui_state()
        
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
            self.display_image(self.images[index])
            
            # Check for existing results
            self.check_for_results()
            
            # Update UI state
            self.update_ui_state()
    
    def on_model_select(self, event):
        """Handle model selection from listbox"""
        selection = self.model_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if 0 <= index < len(self.model_manager.available_models):
            self.selected_model = self.model_manager.available_models[index]
            
            # Check for existing results
            self.check_for_results()
            
            # Update UI state
            self.update_ui_state()
    
    def display_image(self, img_data):
        """Display an image in the original canvas"""
        if img_data is None:
            self.clear_display()
            return
        
        # Get image data
        img = img_data['data']
        
        # Convert BGR to RGB for displaying
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        # Handle initial sizing
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 300
        
        # Calculate scale to fit canvas while maintaining aspect ratio
        img_height, img_width = img.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # Resize the image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Convert to PIL and then to Tkinter PhotoImage
        pil_img = Image.fromarray(resized)
        self.displayed_original = ImageTk.PhotoImage(pil_img)
        
        # Display on canvas
        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.displayed_original
        )
    
    def clear_display(self):
        """Clear both image canvases"""
        # Clear original image
        self.original_canvas.delete("all")
        self.original_canvas.create_text(
            self.original_canvas.winfo_width() // 2,
            self.original_canvas.winfo_height() // 2,
            text="No image selected",
            fill="white",
            font=('Arial', 14)
        )
        
        # Clear processed image
        self.processed_canvas.delete("all")
        self.processed_canvas.create_text(
            self.processed_canvas.winfo_width() // 2,
            self.processed_canvas.winfo_height() // 2,
            text="No processed image",
            fill="white",
            font=('Arial', 14)
        )
    
    def clear_results(self):
        """Clear the results tree"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
    
    def update_ui_state(self):
        """Update the state of UI controls based on current selections"""
        # Check if we have both an image and model selected
        can_analyze = (self.current_image_index >= 0 and 
                       self.selected_model is not None and
                       not self.analyzing)
        
        # Update analyze button state
        if can_analyze:
            self.analyze_button.config(state=tk.NORMAL)
        else:
            self.analyze_button.config(state=tk.DISABLED)
    
    def get_result_path(self, img_name, model_name):
        """Get path for saving/loading results"""
        base_name = os.path.splitext(img_name)[0]
        model_base = os.path.splitext(model_name)[0]
        return os.path.join(self.results_dir, f"{base_name}_{model_base}")
    
    def check_for_results(self):
        """Check if we have existing results for the current image and model"""
        if self.current_image_index < 0 or self.selected_model is None:
            return False
        
        img_name = self.images[self.current_image_index]['name']
        
        # Get result file paths
        json_path = self.get_result_path(img_name, self.selected_model) + ".json"
        img_path = self.get_result_path(img_name, self.selected_model) + ".jpg"
        
        # Check if results exist
        if os.path.exists(json_path) and os.path.exists(img_path):
            try:
                # Load the JSON data
                with open(json_path, 'r') as f:
                    results = json.load(f)
                
                # Load the processed image
                processed_img = cv2.imread(img_path)
                
                # Display results
                self.display_results(processed_img, results)
                return True
            except Exception as e:
                print(f"Error loading existing results: {e}")
        
        return False
    
    def display_results(self, processed_img, results):
        """Display processed image and detection results"""
        if processed_img is None or results is None:
            return
        
        # Display processed image
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.processed_canvas.winfo_width()
        canvas_height = self.processed_canvas.winfo_height()
        
        # Handle initial sizing
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 300
        
        # Calculate scale to fit canvas while maintaining aspect ratio
        img_height, img_width = img_rgb.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # Resize the image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Convert to PIL and then to Tkinter PhotoImage
        pil_img = Image.fromarray(resized)
        self.displayed_processed = ImageTk.PhotoImage(pil_img)
        
        # Display on canvas
        self.processed_canvas.delete("all")
        self.processed_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.displayed_processed
        )
        
        # Update results tree
        self.clear_results()
        
        # Check if this is a classification model
        is_cls_model = results.get("is_classification", False)
        
        if is_cls_model:
            # Classification results
            predictions = results.get("predictions", [])
            for pred in predictions:
                cls_name = pred.get("class", "Unknown")
                confidence = pred.get("confidence", 0) * 100  # Convert to percentage
                
                self.results_tree.insert(
                    "", "end", 
                    values=(cls_name, f"{confidence:.1f}%", "")
                )
        else:
            # Detection results
            detections = results.get("detections", [])
            for det in detections:
                cls_name = det.get("class", "Unknown")
                confidence = det.get("confidence", 0) * 100  # Convert to percentage
                box = det.get("box", [0, 0, 0, 0])
                box_str = f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
                
                self.results_tree.insert(
                    "", "end", 
                    values=(cls_name, f"{confidence:.1f}%", box_str)
                )
    
    def analyze_current(self):
        """Analyze the current image with selected model"""
        if self.analyzing:
            return
            
        if self.current_image_index < 0:
            messagebox.showinfo("No Image", "Please select an image first.")
            return
            
        if self.selected_model is None:
            messagebox.showinfo("No Model", "Please select a model first.")
            return
        
        # Start analysis
        self.analyzing = True
        self.analyze_button.config(state=tk.DISABLED, text="Analyzing...")
        self.status_var.set(f"Analyzing with {self.selected_model}...")
        
        # Show processing indicator
        self.processed_canvas.delete("all")
        self.processed_canvas.create_text(
            self.processed_canvas.winfo_width() // 2,
            self.processed_canvas.winfo_height() // 2,
            text="Analysis in progress...",
            fill="white",
            font=('Arial', 14)
        )
        
        self.clear_results()
        
        # Get the image to analyze
        img_data = self.images[self.current_image_index]
        model_name = self.selected_model
        
        # Run analysis in a thread
        def analysis_thread():
            try:
                # Load model if needed
                if model_name not in self.model_manager.model_instances:
                    self.root.after(0, lambda: 
                        self.status_var.set(f"Loading model {model_name}..."))
                    
                    self.model_manager.load_model(model_name)
                
                # Run inference
                frame = img_data['data'].copy()  # Use a copy to avoid modifying original
                annotated_frame, results, is_cls_model = self.model_manager.predict(
                    frame, model_name
                )
                
                # Prepare result paths
                result_base = self.get_result_path(img_data['name'], model_name)
                json_path = result_base + ".json"
                img_path = result_base + ".jpg"
                
                # Save processed image
                cv2.imwrite(img_path, annotated_frame)
                
                # Extract results data
                json_data = {
                    "model": model_name,
                    "image": img_data['name'],
                    "is_classification": is_cls_model,
                    "processed_image": os.path.basename(img_path)
                }
                
                if is_cls_model and results is not None:
                    # Classification results
                    json_data["predictions"] = []
                    
                    try:
                        # Get predictions
                        probs = results.probs
                        top_indices = probs.top5
                        top_probs = probs.top5conf.cpu().numpy().tolist()
                        names = results.names
                        
                        for i in range(len(top_indices)):
                            cls_idx = int(top_indices[i])
                            prob = float(top_probs[i])
                            cls_name = names[cls_idx]
                            
                            json_data["predictions"].append({
                                "class": cls_name,
                                "confidence": prob
                            })
                    except Exception as e:
                        print(f"Error extracting classification results: {e}")
                
                elif results is not None:
                    # Detection results
                    json_data["detections"] = []
                    
                    try:
                        # Get detections
                        boxes = results.boxes.xyxy.cpu().numpy().tolist()
                        confs = results.boxes.conf.cpu().numpy().tolist()
                        cls_indices = results.boxes.cls.cpu().numpy().tolist()
                        names = results.names
                        
                        for i in range(len(boxes)):
                            box = boxes[i]
                            conf = confs[i]
                            cls_idx = int(cls_indices[i])
                            cls_name = names[cls_idx]
                            
                            json_data["detections"].append({
                                "class": cls_name,
                                "confidence": conf,
                                "box": box
                            })
                    except Exception as e:
                        print(f"Error extracting detection results: {e}")
                
                # Save JSON results
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Update UI with results
                self.root.after(0, lambda: self.display_results(annotated_frame, json_data))
                self.root.after(0, lambda: 
                    self.status_var.set(f"Analysis complete: {model_name}"))
                
            except Exception as e:
                err_msg = str(e)
                print(f"Analysis error: {e}")
                self.root.after(0, lambda err=err_msg: 
                    self.status_var.set(f"Error: {err}"))
                self.root.after(0, lambda err=err_msg: 
                    messagebox.showerror("Analysis Error", f"Error analyzing image: {err}"))
            
            finally:
                self.analyzing = False
                self.root.after(0, lambda: 
                    self.analyze_button.config(state=tk.NORMAL, text="Analyze Image"))
                self.root.after(0, self.update_ui_state)
        
        # Start the thread
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def export_results(self):
        """Export all results to a JSON file"""
        if self.current_image_index < 0:
            messagebox.showinfo("Export Results", "No image selected")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Find all results for the current image
            current_img = self.images[self.current_image_index]
            img_name = current_img['name']
            base_name = os.path.splitext(img_name)[0]
            
            # Build a dictionary with all results for this image
            export_data = {
                "image": img_name,
                "analyses": []
            }
            
            # Check each model for results
            for model in self.model_manager.available_models:
                json_path = self.get_result_path(img_name, model) + ".json"
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            result_data = json.load(f)
                            
                        # Add model results to export data
                        export_data["analyses"].append(result_data)
                    except:
                        pass
            
            # Save combined results
            if export_data["analyses"]:
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                self.status_var.set(f"Results exported to {os.path.basename(file_path)}")
            else:
                messagebox.showinfo("Export Results", "No results found for this image")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")


# Application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOImageAnalyzer(root)
    root.mainloop()