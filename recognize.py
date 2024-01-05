from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from concurrent.futures import ProcessPoolExecutor
from typing import List
import cv2
import shutil
import os
from pathlib import Path
import time
import numpy as np
import torch
from torchvision import transforms
import uvicorn
import sys
from sqlalchemy import null

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "yolov5_face")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from clustering import clustering_design

app = FastAPI()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables to store the detected faces and labels for the previous fully processed frame
prev_frame_faces = []
prev_frame_labels = []

if os.path.exists(r"detected_faces"):
	shutil.rmtree(r"detected_faces")

## Case 2:
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

# Get model recognition
## Case 1: 
from insightface.insight_face import iresnet100
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

isThread = True
score = 0
name = null

# Resize image
def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 256
    conf_thres = 0.7
    iou_thres = 0.9
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

def get_feature(face_image, training = True): 
    # Convert to RGB
    # print("waqar",face_image.shape)
 
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

def read_features(root_fearure_path = "static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def recognition(face_image, images_names, images_embs):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    # isThread = True
    return name, score
# Minimum area threshold to consider a face valid (adjust as needed)
MIN_FACE_AREA = 5000

# (Previous functions)

def main(input_path): 

    Path("detected_faces").mkdir(parents=True, exist_ok=True)
    output_folder = "detected_faces"

    # global isThread, score, name, prev_frame_faces, prev_frame_labels
    # input_path = r"small-video.mp4"
    # scale_factor = 0.5
    
    # Read features
    images_names, images_embs = read_features()
    
    # Open video 
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames of the input video
    frame_count = 0


    # Read until video is completed
    start_total_time = time.time()
    start_time = start_total_time

    # Create a folder to save all detected face images
    detected_folder = os.path.join(output_folder, "detected_faces")

    read_frame_set = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print('Frame#', frame_count, "of", total_frames, "frames")

        # Get faces
        bboxs, landmarks = get_face(frame)

        # Loop through each detected face and save it in the detected images folder
        for i, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            face_image = frame[y1-30:y2+50, x1-30:x2+50]

            # Calculate the area of the bounding box
            bbox_area = (x2 - x1) * (y2 - y1)

            # Check if the face bounding box covers a minimum area
            if bbox_area < MIN_FACE_AREA:
                continue


            # Get recognized name
            if np.any(np.array(face_image.shape) == 0):
                continue
            name, score = recognition(face_image, images_names, images_embs)

            # Save the face image with the name 'detect_{frame_count}_{i}.jpg' inside the detected images folder
            if name == null or score < 0.35:
                label = "Unknown"
            else:
                label = name.split('_')[0]

            face_image_filename = os.path.join(output_folder, f"{label}_{frame_count}.jpg")
            cv2.imwrite(face_image_filename, face_image)

            # Rest of the code for drawing bounding boxes and labels on the frame
            # ...
    read_frame_set += 10

    cap.release()
    # cv2.destroyAllWindows()
    # cv2.waitKey(0)
    print("Detected face images saved in 'detected_faces' folder.")
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    print("Total processing time = ", total_time)
    clustering_design()

def process_video(input_path):
    try:
        main(input_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    video_path = f"uploaded_videos/{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video in a separate thread
    with ProcessPoolExecutor() as executor:
        executor.submit(process_video, video_path)

    return JSONResponse(content={"message": "Video processing started"}, status_code=200)

if __name__ == "__main__":
    #uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run("recognize:app", host="0.0.0.0", port=5004, reload= True)







# if __name__ == "__main__":

#     # Create argument parser
#     parser = argparse.ArgumentParser(description='')
    
#     # Add argument for video path
#     parser.add_argument('video_path',type=str, help='Path to the video file')
    
#     # Parse the command-line arguments
#     args = parser.parse_args()    

#     main(args.video_path)

