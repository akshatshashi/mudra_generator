import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['XFORMERS_DISABLED'] = '1'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk
import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox
from diffusers import StableDiffusionPipeline, AutoencoderKL
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import clip

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
DATASET_PATH = "dataset"
FEATURES_FILE = "mudra_latents.pkl"
IMAGES_PER_GENERATION = 10  # Number of images to generate per click

class OptimizedMudraDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.image_paths = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            images = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths.extend([os.path.join(class_dir, img) for img in images])

        self.transform = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

def extract_latents():
    print("Optimized latent extraction starting...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1", 
        subfolder="vae",
        torch_dtype=torch.float16 if 'cuda' in DEVICE else torch.float32
    ).to(DEVICE)
    vae.eval()

    latent_db = {'train': {}, 'test': {}}

    for phase in ['train', 'test']:
        print(f"\nProcessing {phase} set:")
        dataset = OptimizedMudraDataset(os.path.join(DATASET_PATH, phase))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.no_grad(), tqdm(total=len(dataset), desc=phase) as pbar:
            for batch in loader:
                inputs = batch.to(DEVICE)
                latents = vae.encode(inputs).latent_dist.sample().cpu().numpy()

                for i in range(inputs.size(0)):
                    img_path = dataset.image_paths[pbar.n % len(dataset)]
                    class_name = os.path.basename(os.path.dirname(img_path))
                    
                    if class_name not in latent_db[phase]:
                        latent_db[phase][class_name] = []
                    latent_db[phase][class_name].append(latents[i])
                    
                pbar.update(inputs.size(0))

    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(latent_db, f, protocol=5)
    print("\nLatent extraction completed successfully!")

class MudraGeneratorApp:
    def __init__(self):
        self.latent_db = pickle.load(open(FEATURES_FILE, 'rb'))
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            use_safetensors=True,
            use_xformers=False
        ).to(DEVICE)
        
        self.inception_model = models.inception_v3(pretrained=True)
        self.inception_model.eval().to(DEVICE)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        
        self.generated_images = []
        self.is_scores = []
        self.clip_scores = []
        self.images_per_generation = IMAGES_PER_GENERATION
        
        self.root = tk.Tk()
        self.root.title("Mudra Generator with Evaluation Metrics")
        self.root.geometry("1200x800")
        self._setup_ui()

    def _setup_ui(self):
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Mudra Name:").pack(side=tk.LEFT)
        self.entry = tk.Entry(input_frame, width=30)
        self.entry.pack(side=tk.LEFT, padx=10)
        tk.Button(input_frame, text="Generate", command=self._generate_image).pack(side=tk.LEFT)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        metrics_frame = tk.Frame(self.root)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=metrics_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax1.set_title("Inception Score (Higher is Better)")
        self.ax1.set_xlabel("Number of Generated Images")
        self.ax1.set_ylabel("Score")
        
        self.ax2.set_title("CLIP Score (Higher is Better)")
        self.ax2.set_xlabel("Number of Generated Images")
        self.ax2.set_ylabel("Score")

    def _generate_image(self):
        mudra_name = self.entry.get().strip().title()
        if not self._validate_input(mudra_name):
            return

        try:
            test_latents = self.latent_db['test'][mudra_name]
            
            # Generate multiple images in sequence
            for _ in range(self.images_per_generation):
                latent = torch.from_numpy(
                    test_latents[np.random.randint(len(test_latents))]
                ).to(DEVICE)
                
                generated_image = self.pipe(
                    prompt=f"High quality {mudra_name} mudra hand gesture",
                    latents=latent.unsqueeze(0),
                    guidance_scale=9.0,
                    num_inference_steps=50
                ).images[0]
                
                self.generated_images.append(generated_image)

            # Display last generated image
            self._display_image(generated_image)
            
            # Calculate metrics on all accumulated images
            is_score = self._calculate_inception_score()
            clip_score = self._calculate_clip_score(mudra_name)
            
            # Update scores and plots
            self.is_scores.append(is_score)
            self.clip_scores.append(clip_score)
            self._update_plots()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _validate_input(self, mudra_name):
        if not mudra_name or mudra_name not in self.latent_db['test']:
            messagebox.showerror("Error", "Invalid mudra name or no test samples")
            return False
        return True

    def _display_image(self, image):
        display_img = image.resize((512, 512))
        tk_img = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

    def _calculate_inception_score(self):
        if not self.generated_images:
            return 0.0

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        probs = []
        for img in self.generated_images:
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = self.inception_model(tensor)
            probs.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        marginal = np.mean(probs, axis=0)
        kl_div = np.sum(probs * (np.log(probs) - np.log(marginal)), axis=1)
        return np.exp(np.mean(kl_div))

    def _calculate_clip_score(self, prompt):
        if not self.generated_images:
            return 0.0

        similarities = []
        text_input = clip.tokenize([prompt]).to(DEVICE)
        
        for img in self.generated_images:
            image_input = self.clip_preprocess(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
            similarities.append(torch.cosine_similarity(image_features, text_features).item())
            
        return np.mean(similarities)

    def _update_plots(self):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(range(1, len(self.is_scores)+1), self.is_scores, 'b-o')
        self.ax1.set_title(f"Inception Score: {self.is_scores[-1]:.2f}" if self.is_scores else "Inception Score")
        self.ax1.set_xlabel("Number of Generated Batches")
        self.ax1.set_ylabel("Score")

        self.ax2.plot(range(1, len(self.clip_scores)+1), self.clip_scores, 'r-o')
        self.ax2.set_title(f"CLIP Score: {self.clip_scores[-1]:.2f}" if self.clip_scores else "CLIP Score")
        self.ax2.set_xlabel("Number of Generated Batches")
        self.ax2.set_ylabel("Score")

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    if not os.path.exists(FEATURES_FILE):
        extract_latents()
    
    app = MudraGeneratorApp()
    app.root.mainloop()











import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
os.environ['XFORMERS_DISABLED'] = '1'  # Disable xFormers

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk
import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox
from diffusers import StableDiffusionPipeline, AutoencoderKL
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import clip

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
BATCH_SIZE = 16 if torch.cuda.is_available() else 4  # Larger batch size for GPU, smaller for CPU
DATASET_PATH = "dataset"  # Path to the dataset
FEATURES_FILE = "mudra_latents.pkl"  # File to save/load precomputed latents

# Dataset class for loading and transforming images
class OptimizedMudraDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])  # List of class directories
        self.image_paths = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            images = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # List of image files
            self.image_paths.extend([os.path.join(class_dir, img) for img in images])  # Full paths to images

        self.transform = transforms.Compose([
            transforms.Resize(512, transforms.InterpolationMode.BILINEAR),  # Resize images to 512x512
            transforms.CenterCrop(512),  # Center crop to 512x512
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)  # Return the number of images

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')  # Open image and convert to RGB
        return self.transform(img)  # Apply transformations

# Function to extract latents using a pre-trained VAE
def extract_latents():
    print("Optimized latent extraction starting...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1", 
        subfolder="vae",
        torch_dtype=torch.float16 if 'cuda' in DEVICE else torch.float32  # Use float16 for GPU, float32 for CPU
    ).to(DEVICE)
    vae.eval()  # Set VAE to evaluation mode

    latent_db = {'train': {}, 'test': {}}  # Dictionary to store latents

    for phase in ['train', 'test']:
        print(f"\nProcessing {phase} set:")
        dataset = OptimizedMudraDataset(os.path.join(DATASET_PATH, phase))  # Load dataset
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # Create DataLoader

        with torch.no_grad(), tqdm(total=len(dataset), desc=phase) as pbar:  # No gradient calculation, show progress bar
            for batch in loader:
                inputs = batch.to(DEVICE)  # Move batch to device
                latents = vae.encode(inputs).latent_dist.sample().cpu().numpy()  # Encode images to latents

                for i in range(inputs.size(0)):
                    img_path = dataset.image_paths[pbar.n % len(dataset)]
                    class_name = os.path.basename(os.path.dirname(img_path))
                    
                    if class_name not in latent_db[phase]:
                        latent_db[phase][class_name] = []
                    latent_db[phase][class_name].append(latents[i])  # Store latents in dictionary
                    
                pbar.update(inputs.size(0))  # Update progress bar

    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(latent_db, f, protocol=5)  # Save latents to file
    print("\nLatent extraction completed successfully!")

# GUI application class
class MudraGeneratorApp:
    def __init__(self):
        self.latent_db = pickle.load(open(FEATURES_FILE, 'rb'))  # Load precomputed latents
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            use_safetensors=True,
            use_xformers=False  # Disable xFormers
        ).to(DEVICE)
        
        # Initialize evaluation models
        self.inception_model = models.inception_v3(pretrained=True)  # Load pre-trained Inception v3 model
        self.inception_model.eval().to(DEVICE)  # Set to evaluation mode
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=DEVICE)  # Load CLIP model
        
        # Store generated images and scores
        self.generated_images = []
        self.is_scores = []
        self.clip_scores = []
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Mudra Generator with Evaluation Metrics")
        self.root.geometry("1200x800")
        self._setup_ui()

    def _setup_ui(self):
        # Input frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Mudra Name:").pack(side=tk.LEFT)
        self.entry = tk.Entry(input_frame, width=30)
        self.entry.pack(side=tk.LEFT, padx=10)
        tk.Button(input_frame, text="Generate", command=self._generate_image).pack(side=tk.LEFT)

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Metrics frame
        metrics_frame = tk.Frame(self.root)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=metrics_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot axes
        self.ax1.set_title("Inception Score (Higher is Better)")
        self.ax1.set_xlabel("Number of Generated Images")
        self.ax1.set_ylabel("Score")
        
        self.ax2.set_title("CLIP Score (Higher is Better)")
        self.ax2.set_xlabel("Number of Generated Images")
        self.ax2.set_ylabel("Score")

    def _generate_image(self):
        mudra_name = self.entry.get().strip().title()  # Get mudra name from input
        if not self._validate_input(mudra_name):
            return

        try:
            # Generate image
            test_latents = self.latent_db['test'][mudra_name]
            latent = torch.from_numpy(test_latents[np.random.randint(len(test_latents))]).to(DEVICE)
            
            generated_image = self.pipe(
                prompt=f"High quality {mudra_name} mudra hand gesture",
                latents=latent.unsqueeze(0),
                guidance_scale=9.0,
                num_inference_steps=50
            ).images[0]

            # Store and display image
            self.generated_images.append(generated_image)
            self._display_image(generated_image)
            
            # Calculate metrics
            is_score = self._calculate_inception_score()
            clip_score = self._calculate_clip_score(mudra_name)
            
            # Update scores and plots
            self.is_scores.append(is_score)
            self.clip_scores.append(clip_score)
            self._update_plots()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _validate_input(self, mudra_name):
        if not mudra_name or mudra_name not in self.latent_db['test']:
            messagebox.showerror("Error", "Invalid mudra name or no test samples")
            return False
        return True

    def _display_image(self, image):
        display_img = image.resize((512, 512))  # Resize image for display
        tk_img = ImageTk.PhotoImage(display_img)  # Convert to Tkinter image
        self.image_label.configure(image=tk_img)  # Update image label
        self.image_label.image = tk_img  # Keep reference to avoid garbage collection

    def _calculate_inception_score(self):
        if not self.generated_images:
            return 0.0

        # Preprocess images for Inception v3
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Calculate probabilities
        probs = []
        for img in self.generated_images:
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = self.inception_model(tensor)
            probs.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        marginal = np.mean(probs, axis=0)
        kl_div = np.sum(probs * (np.log(probs) - np.log(marginal)), axis=1)
        return np.exp(np.mean(kl_div))

    def _calculate_clip_score(self, prompt):
        if not self.generated_images:
            return 0.0

        similarities = []
        text_input = clip.tokenize([prompt]).to(DEVICE)
        
        for img in self.generated_images:
            image_input = self.clip_preprocess(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
            similarities.append(torch.cosine_similarity(image_features, text_features).item())
            
        return np.mean(similarities)

    def _update_plots(self):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot Inception Score
        self.ax1.plot(range(1, len(self.is_scores)+1), self.is_scores, 'b-o')
        self.ax1.set_title(f"Inception Score: {self.is_scores[-1]:.2f}" if self.is_scores else "Inception Score")
        self.ax1.set_xlabel("Number of Generated Images")
        self.ax1.set_ylabel("Score")

        # Plot CLIP Score
        self.ax2.plot(range(1, len(self.clip_scores)+1), self.clip_scores, 'r-o')
        self.ax2.set_title(f"CLIP Score: {self.clip_scores[-1]:.2f}" if self.clip_scores else "CLIP Score")
        self.ax2.set_xlabel("Number of Generated Images")
        self.ax2.set_ylabel("Score")

        # Redraw canvas
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    if not os.path.exists(FEATURES_FILE):
        extract_latents()  # Extract latents if not already done
    
    app = MudraGeneratorApp()  # Create and run the GUI application
    app.root.mainloop()
















# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
# os.environ['XFORMERS_DISABLED'] = '1'  

# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image, ImageTk
# import numpy as np
# import pickle
# import tkinter as tk
# from tkinter import messagebox
# from diffusers import StableDiffusionPipeline, AutoencoderKL
# from tqdm import tqdm  


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16 if torch.cuda.is_available() else 4  
# DATASET_PATH = "dataset"
# FEATURES_FILE = "mudra_latents.pkl"


# class OptimizedMudraDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.classes = sorted([d for d in os.listdir(root_dir) 
#                              if os.path.isdir(os.path.join(root_dir, d))])
#         self.image_paths = []
        
        
#         for class_name in self.classes:
#             class_dir = os.path.join(root_dir, class_name)
#             self.image_paths.extend([
#                 os.path.join(class_dir, f) 
#                 for f in os.listdir(class_dir) 
#                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#             ])

        
#         self.transform = transforms.Compose([
#             transforms.Resize(512, transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(512),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5], std=[0.5]),
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img = Image.open(self.image_paths[idx]).convert('RGB')
#         return self.transform(img)

# def extract_latents():
#     print("Optimized latent extraction starting...")
    
    
#     vae = AutoencoderKL.from_pretrained(
#         "stabilityai/stable-diffusion-2-1", 
#         subfolder="vae",
#         torch_dtype=torch.float16 if 'cuda' in DEVICE else torch.float32
#     ).to(DEVICE)
#     vae.eval()

#     latent_db = {'train': {}, 'test': {}}

#     for phase in ['train', 'test']:
#         print(f"\nProcessing {phase} set:")
#         dataset = OptimizedMudraDataset(os.path.join(DATASET_PATH, phase))
#         loader = DataLoader(
#             dataset,
#             batch_size=BATCH_SIZE,
#             shuffle=False,
#             num_workers=0,  
#             pin_memory=True
#         )

#         with torch.no_grad(), tqdm(total=len(dataset), desc=phase) as pbar:
#             for batch in loader:
#                 inputs = batch.to(DEVICE)
#                 latents = vae.encode(inputs).latent_dist.sample().cpu().numpy()

                
#                 for i in range(inputs.size(0)):
#                     img_path = dataset.image_paths[pbar.n % len(dataset)]
#                     class_name = os.path.basename(os.path.dirname(img_path))
                    
#                     if class_name not in latent_db[phase]:
#                         latent_db[phase][class_name] = []
#                     latent_db[phase][class_name].append(latents[i])
                    
#                 pbar.update(inputs.size(0))

    
#     with open(FEATURES_FILE, 'wb') as f:
#         pickle.dump(latent_db, f, protocol=5)
#     print("\nLatent extraction completed successfully!")
    
#     # print("IS score:", calculate_inception_score(latent_db))
#     # priny CLIP score
#     # print("CLIP score:", calculate_clip_score(latent_db))



# class MudraGeneratorApp:
#     def __init__(self):
        
#         self.latent_db = pickle.load(open(FEATURES_FILE, 'rb'))
        
        
#         self.pipe = StableDiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-2-1", 
#             use_safetensors=True,
#             use_xformers=False  
#         ).to(DEVICE)
        
        
#         self.root = tk.Tk()
#         self.root.title("Mudra Diffusion Generator")
#         self.root.geometry("800x600")
#         self.create_widgets()

#     def create_widgets(self):
        
#         self.frame = tk.Frame(self.root, padx=20, pady=20)
#         self.frame.pack()

#         tk.Label(self.frame, text="Enter Mudra Name:").grid(row=0, column=0)
#         self.entry = tk.Entry(self.frame, width=30)
#         self.entry.grid(row=0, column=1)
        
#         tk.Button(self.frame, text="Generate", command=self.generate_image).grid(row=0, column=2)

        
#         self.image_label = tk.Label(self.root)
#         self.image_label.pack()

#     def generate_image(self):
#         mudra_name = self.entry.get().strip().title()
        
#         if not mudra_name or mudra_name not in self.latent_db['test']:
#             messagebox.showerror("Error", "Invalid mudra name or no test samples")
#             return

#         try:
            
#             test_latents = self.latent_db['test'][mudra_name]
#             latent = torch.from_numpy(test_latents[np.random.randint(len(test_latents))]).to(DEVICE)

            
#             image = self.pipe(
#                 prompt=f"High quality {mudra_name} mudra hand gesture",
#                 latents=latent.unsqueeze(0),
#                 guidance_scale=9.0,
#                 num_inference_steps=50
#             ).images[0]

#             self.show_image(image)
            
#         except Exception as e:
#             messagebox.showerror("Generation Error", str(e))

#     def show_image(self, image):
#         image = image.resize((512, 512))
#         tk_image = ImageTk.PhotoImage(image)
#         self.image_label.configure(image=tk_image)
#         self.image_label.image = tk_image

# if __name__ == "__main__":
#     if not os.path.exists(FEATURES_FILE):
#         extract_latents()
    
#     app = MudraGeneratorApp()
#     app.root.mainloop()
