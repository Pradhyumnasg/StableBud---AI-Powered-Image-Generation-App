import tkinter as tk
import customtkinter as ctk 
from PIL import Image
from authtoken import auth_token

import torch
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

# Entry widget
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

# Label to display generated image
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Model setup (Use MPS for MacOS)
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32,  # float32 for MPS compatibility
    use_auth_token=auth_token
) 
pipe.to(device)

def generate(): 
    prompt_text = prompt.get()
    
    # Generate image
    image = pipe(prompt_text, guidance_scale=8.5).images[0]
    
    # Save image
    image.save("generatedimage.png")
    
    # Convert image to CTkImage (Fixed)
    ctk_img = ctk.CTkImage(light_image=Image.open("generatedimage.png"), size=(512, 512))
    
    lmain.configure(image=ctk_img)
    lmain.image = ctk_img  # Prevent garbage collection

# Button
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()
