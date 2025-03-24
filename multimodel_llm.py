import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
import cv2
import random
import glob
import shutil
import threading
import base64
import json
from PIL import Image, ImageTk
from credentials import get_api_key
import google.generativeai as genai

class TrashAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trash Analyzer")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.images = []  # List of {path, name, data}
        self.current_image_index = -1
        self.temp_dir = os.path.join('.', '.temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # API configuration
        self.api_key = get_api_key()
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        self.models = [
            "gemini-2.0-flash",         # Multimodal: text, image, audio, video
            "gemini-2.0-flash-lite",    # Lightweight multimodal
            "gemini-2.0-pro-exp-02-05", # Experimental Pro with multimodal support
            "gemini-1.5-flash",         # Multimodal support
            "gemini-1.5-flash-8b",      # Lightweight multimodal
            "gemini-1.5-pro"            # Multimodal with advanced reasoning
        ]
        self.selected_model = tk.StringVar(value="gemini-1.5-flash")
        self.temperature = tk.DoubleVar(value=0.7)
        self.max_length = tk.IntVar(value=100)
        
        # Create UI layout
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface with the specified layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for prompt
        prompt_frame = ttk.LabelFrame(main_frame, text="Prompt")
        prompt_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prompt_text = scrolledtext.ScrolledText(
            prompt_frame, 
            height=3, 
            wrap=tk.WORD
        )
        self.prompt_text.pack(fill=tk.X, padx=5, pady=5)
        self.prompt_text.insert(tk.END, "I want a short answer for which trash type do you see in the image [cardboard, glass, metal, paper, plastic or other]")
        
        # Middle frame (two panels side by side)
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Left panel - Image list
        left_panel = ttk.LabelFrame(middle_frame, text="Images")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Image control buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Upload Images", command=self.open_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Random Images", command=self.get_random_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Clear Images", command=self.clear_images).pack(side=tk.LEFT, padx=2)
        
        # Listbox for images
        list_frame = ttk.Frame(left_panel)
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
        
        # Right panel - Image display
        right_panel = ttk.LabelFrame(middle_frame, text="Selected Image")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.image_label = ttk.Label(right_panel)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom frame - Configuration and results
        bottom_frame = ttk.LabelFrame(main_frame, text="Configuration and Results")
        bottom_frame.pack(fill=tk.BOTH, pady=(0, 5), ipady=5)
        
        # Split bottom frame into left (config) and right (results)
        bottom_left = ttk.Frame(bottom_frame)
        bottom_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 2.5), pady=5)
        
        bottom_right = ttk.Frame(bottom_frame)
        bottom_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2.5, 5), pady=5)
        
        # Configuration options
        config_frame = ttk.LabelFrame(bottom_left, text="API Configuration")
        config_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selection
        ttk.Label(config_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        model_combo = ttk.Combobox(
            config_frame,
            textvariable=self.selected_model,
            values=self.models,
            state="readonly"
        )
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Temperature slider
        ttk.Label(config_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        temp_slider = ttk.Scale(
            config_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.temperature,
            length=200
        )
        temp_slider.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Create temperature value label with direct reference
        self.temp_value_label = ttk.Label(config_frame, text="0.7")
        self.temp_value_label.grid(row=1, column=2, padx=5)
        
        # Update temperature label when slider moves
        def update_temp_label(*args):
            self.temp_value_label.configure(text=f"{self.temperature.get():.1f}")
        
        self.temperature.trace_add("write", update_temp_label)
        update_temp_label()  # Initialize label
        
        # Max length slider
        ttk.Label(config_frame, text="Max Length:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        length_slider = ttk.Scale(
            config_frame,
            from_=50,
            to=500,
            orient=tk.HORIZONTAL,
            variable=self.max_length,
            length=200
        )
        length_slider.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Create max length value label with direct reference
        self.length_value_label = ttk.Label(config_frame, text="100")
        self.length_value_label.grid(row=2, column=2, padx=5)
        
        # Update length label when slider moves
        def update_length_label(*args):
            self.length_value_label.configure(text=f"{self.max_length.get()}")
        
        self.max_length.trace_add("write", update_length_label)
        update_length_label()  # Initialize label
        
        # Analyze button
        self.analyze_button = ttk.Button(
            config_frame,
            text="Analyze Image",
            command=self.analyze_image
        )
        self.analyze_button.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(config_frame, textvariable=self.status_var, wraplength=250)
        status_label.grid(row=4, column=0, columnspan=3, pady=5)
        
        # Results text area
        results_frame = ttk.LabelFrame(bottom_right, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            height=10,
            font=("TkDefaultFont", 24, "bold"),
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def open_images(self):
        """Open image files from dialog"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        # Clear existing images
        self.clear_images(ask=False)
        
        # Load selected images
        count = 0
        for path in file_paths:
            try:
                # Read image
                img_data = cv2.imread(path)
                if img_data is None:
                    continue
                
                # Convert BGR to RGB for display
                img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                
                # Get file name
                name = os.path.basename(path)
                
                # Add to our list
                self.images.append({
                    'path': path,
                    'name': name,
                    'data': img_data,
                    'display': img_rgb
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
            self._display_current_image()
        
        # Update status
        self.status_var.set(f"Loaded {count} images")
    
    def clear_images(self, ask=True):
        """Clear all loaded images"""
        if ask and self.images:
            if not tk.messagebox.askyesno("Clear Images", "Really clear all images?"):
                return
        
        # Clear image data
        self.images = []
        self.current_image_index = -1
        
        # Clear UI
        self.file_listbox.delete(0, tk.END)
        self.image_label.config(image='')
        self._update_results("")
        
        # Update status
        self.status_var.set("All images cleared")
    
    def get_random_images(self):
        """Get random images from datasets directory"""
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        else:
            # Clear temp directory
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        # Define common trash image locations to search
        search_paths = [
            os.path.join('.', 'datasets', 'trash'),
            os.path.join('.', 'datasets', 'images'),
            os.path.join('.', 'trash_images')
        ]
        
        # Find all image files
        image_files = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, _, files in os.walk(base_path):
                    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                        pattern = os.path.join(root, ext)
                        image_files.extend(glob.glob(pattern))
        
        # If no images found, show error
        if not image_files:
            tk.messagebox.showwarning(
                "No Images Found", 
                "No images found in the datasets directories. Please upload images manually."
            )
            return
        
        # Select up to 20 random images
        selected_images = []
        if len(image_files) > 20:
            selected_images = random.sample(image_files, 20)
        else:
            selected_images = image_files
        
        # Copy to temp directory
        temp_image_paths = []
        for i, img_path in enumerate(selected_images):
            # Get original filename and extension
            orig_name = os.path.basename(img_path)
            extension = os.path.splitext(orig_name)[1]
            
            # Create new filename
            new_name = f"trash_{i+1:02d}{extension}"
            dest_path = os.path.join(self.temp_dir, new_name)
            
            # Copy the file
            try:
                shutil.copy2(img_path, dest_path)
                temp_image_paths.append(dest_path)
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
        
        # Clear current images
        self.clear_images(ask=False)
        
        # Load the temp images
        if temp_image_paths:
            count = 0
            for path in temp_image_paths:
                try:
                    # Read image
                    img_data = cv2.imread(path)
                    if img_data is None:
                        continue
                    
                    # Convert BGR to RGB for display
                    img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                    
                    # Get file name
                    name = os.path.basename(path)
                    
                    # Add to our list
                    self.images.append({
                        'path': path,
                        'name': name,
                        'data': img_data,
                        'display': img_rgb
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
                self._display_current_image()
            
            # Update status
            self.status_var.set(f"Loaded {count} random images")
        else:
            self.status_var.set("No images were loaded")
    
    def on_file_select(self, event):
        """Handle selection from the file listbox"""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if 0 <= index < len(self.images):
            self.current_image_index = index
            self._display_current_image()
    
    def _display_current_image(self):
        """Display the currently selected image"""
        if self.current_image_index < 0 or self.current_image_index >= len(self.images):
            return
        
        # Get image data
        img_rgb = self.images[self.current_image_index]['display']
        
        # Resize image for display while maintaining aspect ratio
        height, width = img_rgb.shape[:2]
        max_height = 400
        max_width = 500
        
        # Calculate new dimensions
        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_height, new_width = int(height * scale), int(width * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update label
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to prevent garbage collection
    
    def analyze_image(self):
        """Analyze the current image using Gemini API"""
        if self.current_image_index < 0:
            tk.messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        # Get current image
        current_image = self.images[self.current_image_index]
        
        # Get prompt text
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            tk.messagebox.showwarning("No Prompt", "Please enter a prompt.")
            return
        
        # Disable analyze button during processing
        self.analyze_button.config(state=tk.DISABLED)
        self.status_var.set("Analyzing image...")
        
        # Clear results
        self._update_results("")
        
        # Run analysis in background thread
        def analysis_thread():
            try:
                # Prepare image for API
                # Convert OpenCV image (BGR) to PIL image (RGB)
                img_rgb = cv2.cvtColor(current_image['data'], cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Get model parameters
                model_name = self.selected_model.get()
                temperature = self.temperature.get()
                max_length = self.max_length.get()
                
                # Create the model with generation config
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_length,
                        "top_p": 0.95,
                        "top_k": 64
                    }
                )
                
                # Generate content using the model with PIL image directly
                response = model.generate_content([prompt, pil_image])
                
                # Process response
                result_text = response.text if hasattr(response, 'text') else str(response)
                self.root.after(0, lambda: self._update_results(result_text))
                self.root.after(0, lambda: self.status_var.set("Analysis complete"))
            
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._update_results(f"ERROR: {error_msg}"))
                self.root.after(0, lambda: self.status_var.set("Error during analysis"))
            
            finally:
                # Enable button again
                self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
        
        # Start thread
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def _update_results(self, text):
        """Update the results text box with bold, larger text"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        
        # Configure tags for bold and larger text
        self.results_text.tag_configure("bold_large", font=("TkDefaultFont", 24, "bold"))
        
        # Apply the tag to all text
        self.results_text.tag_add("bold_large", "1.0", tk.END)
        
        self.results_text.config(state=tk.DISABLED)


# Application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = TrashAnalyzerApp(root)
    root.mainloop()