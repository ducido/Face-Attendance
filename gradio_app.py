import gradio as gr
import cv2
from util import crop_face, clip_recognize
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
yolo_model = YOLO('yolov8n_face.pt')
log_path = "log.txt"

def capture_image():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    success, frame = cap.read()
    if success:
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return rgb_image
    else:
        return None  # return None if no image is captured

def checkin(image):
    cropped_face = crop_face(image, yolo_model, device)
    name = clip_recognize(cropped_face, "db", model, processor, device)
    with open(log_path, 'a') as f:
        f.write('{},{},in\n'.format(name, datetime.datetime.now()))
        f.close()
    return "Welcome {}!".format(name) if name != 'unknown_person' else "Unknown Person, please register first."

def checkout(image):
    # Logic for checking out using the face in the image
    cropped_face = crop_face(image, yolo_model, device)
    name = clip_recognize(cropped_face, "db", model, processor, device)
    with open(log_path, 'a') as f:
        f.write('{},{},out\n'.format(name, datetime.datetime.now()))
        f.close()
    return "Goodbye {}!".format(name) if name != 'unknown_person' else "Unknown Person, please register first."

def register_new_user(image, name):
    if not name:
        return "Please enter your name above", None
    
    # Logic for registering a new user using the face in the image    
    cropped_face = crop_face(image, yolo_model, device)

    # save embedding to db
    inputs = processor(images=cropped_face, return_tensors="pt", padding=True)
    outputs = model.get_image_features(**inputs)
    image_features = outputs / outputs.norm(dim=-1, keepdim=True)
    torch.save(image_features, "db/{}.pt".format(name))
    with open(log_path, 'a') as f:
        f.write('{},{},register\n'.format(name, datetime.datetime.now()))
        f.close()
    return "Registered Successfully, welcome {}!".format(name), cropped_face

# Create Gradio Interface
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            btn_checkin = gr.Button("Checkin")
            btn_checkout = gr.Button("Checkout")
            btn_register = gr.Button("Register New User")
        image = gr.Image(shape=(480, 640), source="webcam", streaming=True, tool="editor")
    with gr.Row():
        with gr.Column():
            name_input = gr.Textbox(label="Enter your name")
            output = gr.Textbox()
        cropped_face = gr.Image(shape=(128, 128), label="Face Cropped")
        
    btn_checkin.click(checkin, inputs=image, outputs=output)
    btn_checkout.click(checkout, inputs=image, outputs=output)
    btn_register.click(register_new_user, inputs=[image, name_input], outputs=[output, cropped_face])

app.launch()
