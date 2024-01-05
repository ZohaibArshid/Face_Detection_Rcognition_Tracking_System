#pytorch
from concurrent.futures import thread
from sqlalchemy import null
import torch
from torchvision import transforms, models
import time
import os
import subprocess
import threading
import queue
# import ffmpeg
import pytz
from collections import defaultdict
import warnings
# Set the time zone to Pakistan Standard Time (PKT)
pakistan_timezone = pytz.timezone('Asia/Karachi')

import sys
import numpy as np
import base64 
import cv2
import pandas as pd
from datetime import datetime
import json
import schedule
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "yolov5_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords



warnings.filterwarnings("ignore", category=UserWarning, message=r".*H.264.*")
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables to store the detected faces and labels for the previous fully processed frame
prev_frame_faces = []
prev_frame_labels = []
images_names = []
images_embs = []
output_dir = "output_videos"
audio_dir = "audio_chunks"
person_data = []
json_data = None
face_detected_data = []
saved_faces = {}
unknown_query_embs = {}
unknown_count = 0

#model = attempt_load("scripts/yolov5_face/yolov5m-face.pt", map_location=device)
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

# Get model recognition 
from insightface.insight_face import iresnet100
#weight = torch.load("scripts/insightface/resnet100_backbone.pth", map_location = device)
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

#model_emb = models.resnet50(pretrained=True)
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

# isThread = True
score = 0
name = null


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
    size_convert = 1056
    conf_thres = 0.75
    iou_thres = 0.75
    
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
    return name, score, query_emb

def time_str(total_seconds):
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return timestamp_str

def time_to_seconds(timestamp_str):
    try:
        hours, minutes, seconds = map(int, timestamp_str.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except ValueError:
        raise ValueError("Invalid timestamp format. Use hh:mm:ss")
    
def numpy_array_to_base64(image_array, format='.jpg'):
    _, buffer = cv2.imencode(format, image_array)
    base64_image = base64.b64encode(buffer).decode()
    return base64_image

def remove_unknown_entries():
    data = np.load('static/feature/face_features.npz')

    arr1 = data['arr1']
    arr2 = data['arr2']
    # Create empty lists to store updated data
    updated_arr1 = []
    updated_arr2 = []

    for label, emb in zip(arr1, arr2):
        if not label.startswith('Unknown'):
            updated_arr1.append(label)
            updated_arr2.append(emb)

    # Convert the updated lists to numpy arrays
    updated_arr1 = np.array(updated_arr1)
    updated_arr2 = np.array(updated_arr2)

    # Save the updated data back to the "face_features.npz" file
    np.savez("static/feature/face_features.npz", arr1=updated_arr1, arr2=updated_arr2)

def get_unknown_count():
    if os.path.isfile('unknown_count.txt'):
        with open('unknown_count.txt', 'r') as count_file:
            unknown_count = int(count_file.read())
        return unknown_count
    else:
        unknown_count = 0
        return unknown_count

def update_unknown_count(new_count):
    with open('unknown_count.txt', 'w') as count_file:
        count_file.write(str(new_count))

def update_npz_file(unknown_query_embs):
    data = np.load('static/feature/face_features.npz')
    arr1 = data['arr1']
    arr2 = data['arr2']

    # Create empty lists to store updated data
    updated_arr1 = []
    updated_arr2 = []

    # Add the existing data to the updated lists
    for key, value in unknown_query_embs.items():
        if key not in arr1:
            updated_arr1.extend([key] * len(value))
            updated_arr2.extend(value)

    # Convert the updated lists to numpy arrays
    updated_arr1 = np.array(updated_arr1)
    updated_arr2 = np.array(updated_arr2)

    # Combine the updated data with the existing data
    if arr1 is not None and arr2 is not None:
        updated_arr1 = np.concatenate((arr1, updated_arr1))
        #updated_arr2 = np.concatenate((arr2, updated_arr2))
        if updated_arr2.size == 0:
            updated_arr2 = arr2
        else:
            updated_arr2 = np.vstack((arr2, updated_arr2))

    # Save the updated data back to the "face_features.npz" file
    np.savez("static/feature/face_features.npz", arr1=updated_arr1, arr2=updated_arr2)

def processing_chunk(chunk_timestamp):
    global person_data,json_data

    #cap = cv2.VideoCapture("rtsp://admin:hik12345@10.0.41.161:554/")
    cap = cv2.VideoCapture(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    # video = cv2.VideoWriter(output_without_audio_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, size)
    frame_interval = int(output_fps / 3)
    #frame_interval = 15
    prev_frame_faces, prev_frame_labels = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            continue
        if not ret:
            break

        frame_count += 1
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_seconds = int((position_ms / 1000) + time_to_seconds(chunk_timestamp))
        frame_timestamp = time_str(timestamp_seconds)

        if frame_count % frame_interval != 0 and frame_count != 1:
            for box, label in zip(prev_frame_faces, prev_frame_labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            #video.write(frame)
            continue

        try:
            bboxs, landmarks = get_face(frame)
        except cv2.error as e:
            print(f"Error in get_face(): {e}. Skipping frame.")
            continue
        
        prev_frame_faces = []
        prev_frame_labels = []
        unknown_persons = []

        unknown_count = get_unknown_count()

        for i in range(len(bboxs)):

            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            face_image = frame[y1:y2, x1:x2]

            images_names, images_embs = read_features()
            
            name, score, query_emb = recognition(face_image, images_names, images_embs)

            if name is None:
                continue
            else:
                if score < 0.35:
                    existing_labels = list(unknown_query_embs.keys())
                    existing_label = next((label for label in existing_labels if label.startswith("Unknown") and unknown_query_embs[label] is None), None)

                    if existing_label:
                        label = existing_label
                    else:
                        unknown_count += 1
                        label = f"Unknown{unknown_count}"
                        unknown_query_embs[label] = query_emb
                        
                    unknown_persons.append({
                        'query_emb' : query_emb,
                        'label': label,
                        'bbox': bboxs[i],
                        'score': score,
                        'frame_timestamp': frame_timestamp
                    })

                else:
                    label = name.replace("_", " ")

                if label.startswith("Unknown"):
                    # For unknown persons, create separate entries
                    person_entry = {
                        'name': label,
                        'thumbnail': None,
                    }
                    person_data.append(person_entry)
                else:
                    # Find the person in the DataFrame or create a new entry
                    person_entry = next((p for p in person_data if p['name'] == label), None)
                    if person_entry is None:
                        person_entry = {
                            'name': label,
                            'thumbnail': None,
                        }
                        person_data.append(person_entry)
       
                # Update the person's data
                if person_entry['thumbnail'] is None:
                    person_entry['thumbnail'] = numpy_array_to_base64(face_image)

                caption = f"{label}"
                prev_frame_labels.append(label)
                prev_frame_faces.append(bboxs[i])
                t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        update_npz_file(unknown_query_embs)
        update_unknown_count(unknown_count)
        # output_folder = "output_frame"
        # frame_name = "live.png"
        # frame_path = os.path.join(output_folder, frame_name)
        # cv2.imwrite(frame_path, frame)
        cv2.imshow('Video Frame', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        #video.write(frame)

    #video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

    df = pd.DataFrame(person_data)

    filtered_df = df

    max_unknown_number = 0
    for name in filtered_df['name']:
        if name.startswith('Unknown'):
            number = int(name[len('Unknown'):])
            max_unknown_number = max(max_unknown_number, number)

    # Save the maximum unknown number to a text file
    with open('unknown_count.txt', 'w') as max_unknown_file:
        max_unknown_file.write(str(max_unknown_number))


    file_path = "data.json"
    json_data = filtered_df.to_json(orient='records')
    
    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)
    print("Data saved to:", file_path)

    if filtered_df is not None:
        output_csv_path = f"output_videos.csv"
        #filtered_df_copy = filtered_df.drop('thumbnail', axis=1)
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"DataFrame saved to '{output_csv_path}'.")
    else:
        print("No person of interest detected!")

def main_func():
    global images_names, images_embs, output_dir, audio_dir,video_chunk_dir

    # Read features
    images_names, images_embs = read_features()
    print("Read features successful")

    # Create a list of timestamps for person data
    label_names = list(set(images_names))
    for n in label_names:
        n = n.replace("_", " ")
        person_entry = {
            'thumbnail': None,
            'name': n,
            'timestamps': [],
            'coverageTime': '00:00:00'
        }
        person_data.append(person_entry)

    chunk_timestamp = "00:00:00"  # Replace with the actual timestamp

    processing_start = time.time()

    processing_chunk(chunk_timestamp)

    processing_end = time.time()
    total_processing_time = processing_end - processing_start
    print("Chunk processing time: ", total_processing_time)

if __name__=="__main__":
    main_func()
