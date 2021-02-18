# from keras.models import load_model
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import PIL.ImageOps
import json
import numpy as np
import base64
from io import BytesIO
import requests

API_GATEWAY_URL = "inference-cleco-mnist-demo.default-tenant.app.seagate-demo-cluster.iguazio-cd2.com"
model_url = f"http://{API_GATEWAY_URL}/v2/models/model/predict"


def encode_image(img):
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64.decode("utf-8")


def predict_digit(img):
    payload = json.dumps({"inputs": [encode_image(img)]})
    response = requests.post(model_url, json=payload)
    predictions = response.json()
    digit, acc = predictions["outputs"]
    return digit, acc


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognise", command=self.classify_handwriting
        )
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(
            row=0,
            column=0,
            pady=2,
            sticky=W,
        )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        x, y, w, h = rect
        # a, b, c, d = rect
        # rect = (a + 4, b + 4, c - 4, d - 4)
        rect = (x + 40, y + 40, w + 40, h + 40)
        im = ImageGrab.grab(rect)
        im = PIL.ImageOps.invert(im)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ", " + str(int(acc * 100)) + "%")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(
            self.x - r, self.y - r, self.x + r, self.y + r, fill="black"
        )


app = App()
mainloop()
