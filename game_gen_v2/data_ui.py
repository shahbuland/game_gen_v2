import tkinter as tk
import subprocess

import os
import json
from tkinter import filedialog, messagebox

from game_gen_v2.data.pipelines.embed_walk import walking_embed, clear_embeds
from game_gen_v2.data.pipelines.make_input_tensors import gen_input_tensors, clear_controls
from game_gen_v2.data.pipelines.generate_train_set import generate_train_set, clear_dataset
from game_gen_v2.data.pipelines.diversity_measures import process_control_files

class DataProcessingUI:
    def __init__(self, master):
        self.master = master
        master.title("Data Processing UI")

        # Load default config
        self.config = self.load_default_config()

        # Create and pack widgets
        self.create_config_widgets()
        self.create_action_buttons()

    def load_default_config(self):
        config = {
            "VAE_BATCH_SIZE": 64,
            "LATENT": False,
            "ASSUMED_SHAPE": (3, 128, 128),
            "VAE_PATH": "madebyollin/taesdxl",
            "OUT_H": 128,
            "OUT_W": 128,
            "FPS_IN": 60,
            "FPS_OUT": 60,
            "SEGMENT_LENGTH": 500,
            "KEYBINDS": ["SPACE", "W", "A", "S", "D","LSHIFT"],
            "IN_DIR": "E:/data_dump/games",
            "OUT_DIR": "E:/datasets/128_60fps_wasd_many"
        }
        return config

    def create_config_widgets(self):
        for key, value in self.config.items():
            frame = tk.Frame(self.master)
            frame.pack(fill=tk.X, padx=5, pady=5)

            label = tk.Label(frame, text=key, width=20, anchor='w')
            label.pack(side=tk.LEFT)

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                entry = tk.Checkbutton(frame, variable=var)
            elif isinstance(value, tuple):
                var = tk.StringVar(value=str(value))
                entry = tk.Entry(frame, textvariable=var)
            elif isinstance(value, list):
                var = tk.StringVar(value=", ".join(value))
                entry = tk.Entry(frame, textvariable=var)
            else:
                var = tk.StringVar(value=str(value))
                entry = tk.Entry(frame, textvariable=var)

            entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
            setattr(self, f"{key}_var", var)

    def create_action_buttons(self):
        actions_frame = tk.Frame(self.master)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)

        embed_frame = tk.Frame(actions_frame)
        embed_frame.pack(side=tk.LEFT, padx=5)
        tk.Button(embed_frame, text="Embed Walk", command=self.run_embed_walk).pack(side=tk.LEFT)
        tk.Button(embed_frame, text="Clear Embeddings", command=self.clear_embeddings).pack(side=tk.LEFT)

        tk.Button(actions_frame, text="Make Input Tensors", command=self.run_make_input_tensors).pack(side=tk.LEFT, padx=5)
        tk.Button(actions_frame, text="Clear Input Tensors", command=self.clear_input_tensors).pack(side=tk.LEFT, padx=5)
        tk.Button(actions_frame, text="Generate Train Set", command=self.run_generate_train_set).pack(side=tk.LEFT, padx=5)
        tk.Button(actions_frame, text="Clear Train Set", command=self.clear_train_set).pack(side=tk.LEFT, padx=5)
        tk.Button(actions_frame, text="Diversity Measures", command=self.run_diversity_measures).pack(side=tk.LEFT, padx=5)
        tk.Button(actions_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)

    def update_config(self):
        for key in self.config.keys():
            var = getattr(self, f"{key}_var")
            if isinstance(self.config[key], bool):
                self.config[key] = var.get()
            elif isinstance(self.config[key], tuple):
                self.config[key] = eval(var.get())
            elif isinstance(self.config[key], list):
                self.config[key] = [item.strip() for item in var.get().split(",")]
            else:
                self.config[key] = type(self.config[key])(var.get())

    def save_config(self):
        self.update_config()
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully!")

    def run_embed_walk(self):
        self.update_config()
        walking_embed(
            self.config['VAE_BATCH_SIZE'],
            self.config['FPS_IN'] // self.config['FPS_OUT'],
            self.config['OUT_H'],
            self.config['OUT_W'],
            self.config['IN_DIR'],
            self.config['LATENT']
        )
        messagebox.showinfo("Success", "Embed walk completed successfully!")

    def run_make_input_tensors(self):
        self.update_config()
        gen_input_tensors(
            self.config['IN_DIR'],
            self.config['FPS_OUT'],
            self.config['KEYBINDS']
        )
        messagebox.showinfo("Success", "Input tensors generated successfully!")

    def run_generate_train_set(self):
        self.update_config()
        generate_train_set(
            self.config['IN_DIR'],
            self.config['OUT_DIR'],
            self.config['LATENT'],
            self.config['SEGMENT_LENGTH'],
            self.config['ASSUMED_SHAPE']
        )
        messagebox.showinfo("Success", "Train set generated successfully!")

    def run_diversity_measures(self):
        self.update_config()
        process_control_files(self.config['OUT_DIR'])
        messagebox.showinfo("Success", "Diversity measures calculated successfully!")

    def clear_embeddings(self):
        self.update_config()
        clear_embeds(self.config["IN_DIR"], self.config["LATENT"])
        messagebox.showinfo("Success", "Embeddings cleared successfully!")

    def clear_input_tensors(self):
        self.update_config()
        clear_controls(self.config["IN_DIR"])
        messagebox.showinfo("Success", "Input tensors cleared successfully!")

    def clear_train_set(self):
        self.update_config()
        clear_dataset(self.config["OUT_DIR"])
        messagebox.showinfo("Success", "Train set cleared successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataProcessingUI(root)
    root.mainloop()
