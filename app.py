import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk

from face_engine import FaceEngine
from trainer import rebuild_model, load_centroids, predict_name
from utils_fs import ensure_person_dir

# Force CPU = -1, use GPU if available = 0
CTX_ID = 0


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition – ArcFace (InsightFace)")
        self.root.geometry("700x550")

        self.status_var = tk.StringVar(value="Ready.")

        title = tk.Label(root, text="Face Recognition (ArcFace)", font=("Segoe UI", 16, "bold"))
        title.pack(pady=8)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_add = tk.Button(btn_frame, text="Add Person", width=16, command=self.on_add_person)
        self.btn_add.grid(row=0, column=0, padx=8)

        self.btn_test = tk.Button(btn_frame, text="Test (Camera)", width=16, command=self.on_test)
        self.btn_test.grid(row=0, column=1, padx=8)

        self.btn_stop = tk.Button(btn_frame, text="Stop Test", width=16, command=self.stop_test, state="disabled")
        self.btn_stop.grid(row=0, column=2, padx=8)

        self.btn_rebuild = tk.Button(btn_frame, text="Rebuild Model", width=16, command=self.on_rebuild)
        self.btn_rebuild.grid(row=1, column=0, columnspan=3, pady=8)

        self.status = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status.pack(fill="x", padx=12, pady=6)

        # Video preview area
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Camera / capture variables
        self.cap = None
        self.capture_name = None
        self.capture_dir = None
        self.capture_count = 0
        self.max_images = 3
        self.testing = False  # Flag for camera test

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ------------------- Add Person -------------------
    def on_add_person(self):
        name = simpledialog.askstring("Add Person", "Enter person's name:")
        if not name:
            return
        self.capture_dir = ensure_person_dir(name)
        self.capture_name = name
        self.capture_count = 0

        self.set_status(f"Capturing {self.max_images} images for {name}...")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        self.update_frame()

    # def update_frame(self):
    #     if not self.cap or not self.cap.isOpened():
    #         return

    #     ret, frame = self.cap.read()
    #     if not ret:
    #         self.root.after(10, self.update_frame)
    #         return

    #     # Show frame in Tkinter
    #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(rgb)
    #     imgtk = ImageTk.PhotoImage(image=img)
    #     self.video_label.imgtk = imgtk
    #     self.video_label.configure(image=imgtk)

    #     # Save images (one per second)
    #     if self.capture_count < self.max_images:
    #         now = time.time()
    #         if not hasattr(self, "_last_shot") or now - self._last_shot > 1.0:
    #             filename = os.path.join(self.capture_dir, f"{self.capture_name}_{self.capture_count+1}.jpg")
    #             cv2.imwrite(filename, frame)
    #             self.capture_count += 1
    #             self._last_shot = now
    #             self.set_status(f"Captured image {self.capture_count}/{self.max_images} for {self.capture_name}")

    #     if self.capture_count < self.max_images:
    #         self.root.after(30, self.update_frame)
    #     else:
    #         self.set_status(f"Done capturing images for {self.capture_name}")
    #         self.cap.release()
    #         self.cap = None


    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        # Show frame in Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Only capture image if a face is detected
        fe = FaceEngine(ctx_id=CTX_ID)
        face = fe.get_best_face(frame)
        if face is not None:
            if self.capture_count < self.max_images:
                now = time.time()
                if not hasattr(self, "_last_shot") or now - self._last_shot > 1.0:
                    filename = os.path.join(self.capture_dir, f"{self.capture_name}_{self.capture_count+1}.jpg")
                    cv2.imwrite(filename, frame)
                    self.capture_count += 1
                    self._last_shot = now
                    self.set_status(f"Captured image {self.capture_count}/{self.max_images} for {self.capture_name}")

        if self.capture_count < self.max_images:
            self.root.after(30, self.update_frame)
        else:
            self.set_status(f"Done capturing images for {self.capture_name}")
            self.cap.release()
            self.cap = None


    # ------------------- Camera Test -------------------
    def on_test(self):
        self.set_status("Testing with camera...")
        fe = FaceEngine(ctx_id=CTX_ID)
        centroids, meta = load_centroids()
        if not centroids:
            messagebox.showwarning("No Model", "No trained model found. Please rebuild model first.")
            self.set_status("No model available.")
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        self.testing = True
        self.btn_stop.config(state="normal")
        self.set_status("Testing with camera... (click Stop to exit)")

        def update_test_frame():
            if not self.testing or not self.cap.isOpened():
                return

            ret, frame = self.cap.read()
            if not ret:
                self.set_status("Failed to read from camera")
                self.cap.release()
                return

            # Face recognition
            face = fe.get_best_face(frame)
            if face is not None and face.normed_embedding is not None:
                emb = face.normed_embedding.astype(np.float32)
                name, sim = predict_name(emb, centroids, meta.get("threshold_sim", 0.38))
                fe.draw_bbox_with_label(frame, face.bbox, f"{name} ({sim:.2f})")

            # Convert BGR -> RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Repeat after 30 ms
            if self.testing:
                self.root.after(30, update_test_frame)

        update_test_frame()

    def stop_test(self):
        self.testing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.video_label.configure(image=None)
        self.btn_stop.config(state="disabled")
        self.set_status("Stopped camera test.")

    # ------------------- Rebuild Model -------------------
    def on_rebuild(self):
        self.set_status("Rebuilding model...")
        result = rebuild_model(ctx_id=CTX_ID)
        messagebox.showinfo("Model Rebuilt", f"Classes: {result['classes']}")
        self.set_status("Model rebuilt successfully.")


    

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
