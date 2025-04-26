import os
import uuid
from datetime import datetime
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.menu import MDDropdownMenu
from kivy.uix.image import AsyncImage
from kivy.core.window import Window

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

Window.size = (600, 700)

class MainApp(MDApp):
    def build(self):
        # Light theme
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"

        # Device and models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = {
            "Stable Diffusion v1.4": "CompVis/stable-diffusion-v1-4",
            "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
            "Dreamlike Photoreal 2.0": "dreamlike-art/dreamlike-photoreal-2.0",
        }

        # Load default model
        self.selected_model_name = "Stable Diffusion v1.4"
        self.current_model_id = self.available_models[self.selected_model_name]
        self.pipe = self.load_model(self.current_model_id)

        # UI layout
        self.layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)

        # Prompt input
        self.prompt_input = MDTextField(
            hint_text="Enter your prompt...",
            size_hint=(1, None),
            height=60,
            mode="rectangle"
        )
        self.layout.add_widget(self.prompt_input)

        # Dropdown menu items
        menu_items = [
            {
                "text": name,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=name: self.set_model(x)
            }
            for name in self.available_models
        ]

        # Model dropdown
        self.model_menu = MDDropdownMenu(
            caller=None,
            items=menu_items,
            width_mult=4
        )
        self.model_button = MDFlatButton(
            text=self.selected_model_name,
            on_release=lambda _: self.model_menu.open()
        )
        self.model_menu.caller = self.model_button
        self.layout.add_widget(self.model_button)

        # Generate button
        self.generate_button = MDRaisedButton(
            text="Generate",
            size_hint=(1, None),
            height=50,
            on_release=self.generate_image
        )
        self.layout.add_widget(self.generate_button)

        # Image output display
        self.image_widget = AsyncImage(
            source='',
            size_hint=(1, 0.85),
            allow_stretch=True,
            keep_ratio=True
        )
        self.layout.add_widget(self.image_widget)

        return self.layout

    def load_model(self, model_id):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision="main",
            torch_dtype=torch.float16,
            use_auth_token=auth_token
        )
        pipe.to(self.device)
        return pipe

    def set_model(self, model_name):
        self.selected_model_name = model_name
        self.model_button.text = model_name
        self.current_model_id = self.available_models[model_name]
        self.model_menu.dismiss()
        self.pipe = self.load_model(self.current_model_id)

    def generate_image(self, instance):
        prompt = self.prompt_input.text.strip()
        if not prompt:
            return

        with autocast(self.device):
            print(self.device)
            image = self.pipe(prompt, guidance_scale=15).images[0]

        # Create model folder if it doesn't exist
        model_folder = f"generated_images/{self.selected_model_name.replace(' ', '_')}"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Generate a unique filename
        unique_filename = f"{str(uuid.uuid4())}.png"

        # Full path to save the image
        image_path = os.path.join(model_folder, unique_filename)

        # Save the image
        image.save(image_path)

        # Display image
        self.image_widget.source = image_path
        self.image_widget.reload()

if __name__ == '__main__':
    MainApp().run()
