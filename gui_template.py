import tkinter as tk
from tkinter import font, Canvas, Scrollbar
from datetime import datetime
import threading
from PIL import Image, ImageTk
import requests
from io import BytesIO

class ChatGUI:
    def __init__(self, window_title="💧 Water Assistant", on_submit=None):
        self.on_submit = on_submit

        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.geometry("400x750")
        self.window.configure(bg="#0e0e0e")
        self.window.resizable(False, False)

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=11)

        # Chat log
        self.chat_log_frame = tk.Frame(self.window, bg="#1a1a1a")
        self.chat_log_frame.pack(fill="both", expand=True, padx=8, pady=(10, 0))

        self.chat_log_canvas = Canvas(self.chat_log_frame, bg="#1a1a1a", highlightthickness=0)
        self.scrollbar = Scrollbar(self.chat_log_frame, command=self.chat_log_canvas.yview)
        self.chat_log_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.chat_log_canvas.pack(side="left", fill="both", expand=True)

        self.chat_log_inner = tk.Frame(self.chat_log_canvas, bg="#1a1a1a")
        self.chat_window = self.chat_log_canvas.create_window((0, 0), window=self.chat_log_inner, anchor='nw')
        self.chat_log_inner.bind("<Configure>", self.on_configure)

        # Input
        self.input_container = tk.Frame(self.window, bg="#0e0e0e", pady=10)
        self.input_container.pack(fill="x", padx=10)

        self.user_input = tk.Entry(self.input_container, font=("Segoe UI", 12), bg="#292929", fg="white", insertbackground="white", relief="flat")
        self.user_input.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.submit_question())

        self.send_btn = tk.Button(self.input_container, text="Send", command=self.submit_question, bg="#25D366", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=6, cursor="hand2")
        self.send_btn.pack(side="right")

        self.typing_label = None
        self.insert_welcome_message()

    def run(self):
        self.window.mainloop()

    def on_configure(self, event):
        self.chat_log_canvas.configure(scrollregion=self.chat_log_canvas.bbox("all"))

    def submit_question(self):
        question = self.user_input.get().strip()
        if not question:
            return
        timestamp = datetime.now().strftime("%H:%M")
        self.insert_message("You", question, "#25D366", anchor="e", status="✓✓", timestamp=timestamp)
        self.user_input.delete(0, tk.END)
        self.show_typing_indicator()
        if self.on_submit:
            threading.Thread(target=self.on_submit, args=(question,), daemon=True).start()

    def display_reply(self, reply):
        self.remove_typing_indicator()
        timestamp = datetime.now().strftime("%H:%M")
        self.insert_message("Bot", reply, "#444444", anchor="w", timestamp=timestamp)

    def insert_message(self, sender, message, bg_color, anchor="w", status="", timestamp=""):
        bubble_frame = tk.Frame(self.chat_log_inner, bg="#1a1a1a", padx=6, pady=3)
        msg_label = tk.Label(
            bubble_frame, text=message, bg=bg_color, fg="white",
            font=("Segoe UI", 11), wraplength=280, justify="left", padx=10, pady=6
        )
        msg_label.pack(anchor=anchor)

        status_text = f"{timestamp} {status}".strip()
        info_label = tk.Label(
            bubble_frame, text=status_text, bg=bg_color, fg="#cccccc",
            font=("Segoe UI", 8), justify="right"
        )
        info_label.pack(anchor=anchor, padx=10, pady=(0, 4))
        bubble_frame.pack(anchor=anchor, fill="x", pady=4, padx=8)
        self.chat_log_canvas.update_idletasks()
        self.chat_log_canvas.yview_moveto(1.0)

    def show_typing_indicator(self):
        self.typing_label = tk.Label(self.chat_log_inner, text="Bot is typing...", fg="#bbbbbb", bg="#1a1a1a", font=("Segoe UI", 10, "italic"))
        self.typing_label.pack(anchor="w", padx=10, pady=5)
        self.chat_log_canvas.update_idletasks()
        self.chat_log_canvas.yview_moveto(1.0)

    def remove_typing_indicator(self):
        if self.typing_label:
            self.typing_label.destroy()
            self.typing_label = None

    def insert_welcome_message(self):
        welcome = """👋 Hello! I'm your Water Treatment Expert Assistant.

🔹 I can help you with:
• Product recommendations (RO, UV, UF systems)
• Water science education (alkaline water, TDS, pH)
• Technology comparisons and benefits
• Finding the right system for your needs

💬 Just ask me anything about water treatment or say hello to get started!"""
        timestamp = datetime.now().strftime("%H:%M")
        self.insert_message("Bot", welcome, "#444444", anchor="w", timestamp=timestamp)

    def display_image(self, image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            image.thumbnail((280, 280))
            photo = ImageTk.PhotoImage(image)

            image_label = tk.Label(self.chat_log_inner, image=photo, bg="#1a1a1a")
            image_label.image = photo
            image_label.pack(anchor="w", padx=10, pady=6)

            self.chat_log_canvas.update_idletasks()
            self.chat_l_
        except Exception as e:
            print(f"Error displaying image: {e}")