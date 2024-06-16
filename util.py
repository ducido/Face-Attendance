import os
import torch

def clip_recognize(img, db_path, clip_model, processor, device):
    # it is assumed there will be at most 1 match in the db
    img = torch.tensor(img).to(device)
    img = processor(images=img, return_tensors="pt", padding=True)
    img['pixel_values'] = img['pixel_values'].to(device)
    embeddings_unknown = clip_model.get_image_features(**img)
    embeddings_unknown = embeddings_unknown / embeddings_unknown.norm(dim=-1, keepdim=True)

    db_dir = sorted(os.listdir(db_path))

    match = False
    j = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])
        embeddings = torch.load(path_).to(device)

        match = torch.matmul(embeddings_unknown, embeddings.T) > 0.6
        j += 1

    if match:
        return db_dir[j - 1].split('.')[0]
    else:
        return 'unknown_person'


def yolo_detection(img, yolo_model, device):
    # img = torch.tensor(img).to(device)
    pred = yolo_model(img)
    pred = pred[0]
    return pred

def crop_face(img, yolo_model, device):
    pred = yolo_detection(img, yolo_model, device)
    bboxes = pred.boxes.xyxy
    conf = pred.boxes.conf

    # Define a threshold for confidence
    conf_threshold = 0.2

    # List to store cropped face images
    cropped_faces = []

    # Process each detected bounding box
    for i in range(len(bboxes)):
        if conf[i] >= conf_threshold:
            x1, y1, x2, y2 = bboxes[i].int().tolist()

            # Crop the tensor
            cropped_img = img[y1:y2, x1:x2]
            cropped_faces.append(cropped_img)
    arg_max = torch.argmax(conf)        
    print('total face', len(cropped_faces))
    return cropped_faces[arg_max]
