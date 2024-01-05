#pytorch
import torch
from torchvision import transforms

#other lib
import sys
import numpy as np
import os
import cv2
import time
from sort import Sort

sys.path.insert(0, "yolov5_face")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detect
model = attempt_load("yolov5m-face.pt", map_location=device)
face_id_counter = 1
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
    size_convert = 128
    conf_thres = 0.4
    iou_thres = 0.5
    
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

def main():
    global next_id, face_mapping, face_id_counter
    # Open camera 
    cap = cv2.VideoCapture(0)
    start = time.time_ns()
    frame_count = 0
    fps = -1
    
    # Save video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    #video = cv2.VideoWriter('results/face-detection.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    # Initialize SORT tracker
    sort_tracker = Sort()
    while(True):
        face_id_counter = 1
        _, frame = cap.read()
        # Get faces
        bboxs, landmarks = get_face(frame)

        # Apply SORT tracker
        dets_to_sort = np.empty((0, 5))
        face_mapping = {}
        if bboxs is not None:
            for bbox in bboxs:
                x1, y1, x2, y2 = bbox
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, 0])))  # 0 is the default class ID
                print("dets_to_sort",dets_to_sort)
            # Run SORT tracker
            tracks = sort_tracker.update(dets_to_sort)
            #tracks = sort_tracker.getTracks()
            print("tracks", tracks)

            # Visualize tracks on the image
            for track in tracks:
                print("track",track)
                track_id = int(track[-1])  # Extract track ID from the last element
                print("track_id", track_id)
                if track_id not in face_mapping:
                    face_mapping[track_id] = face_id_counter
                    face_id_counter += 1

                face_id = face_mapping[track_id]
                print("face_id", face_id)

                color = (int(face_id * 255 / (len(face_mapping) + 1)) % 255, 255, 255)
                # Draw bounding box
                cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 2)

                # Draw track ID
                cv2.putText(frame, str(face_id), (int(track[0]), int(track[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print("len(face_mapping)",len(face_mapping))

        # Show result
        cv2.imshow("Face Detection", frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

if __name__ == "__main__":
    main()































# #pytorch
# import torch
# from torchvision import transforms

# #other lib
# import sys
# import numpy as np
# import os
# import cv2
# import time

# sys.path.insert(0, "yolov5_face")

# from models.experimental import attempt_load
# from utils.datasets import letterbox
# from utils.general import check_img_size, non_max_suppression_face, scale_coords

# # Check device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Get model detect
# model = attempt_load("yolov5m-face.pt", map_location=device)

# # Resize image
# def resize_image(img0, img_size):
#     h0, w0 = img0.shape[:2]  # orig hw
#     r = img_size / max(h0, w0)  # resize image to img_size

#     if r != 1:  # always resize down, only resize up if training with augmentation
#         interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
#         img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

#     imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
#     img = letterbox(img0, new_shape=imgsz)[0]

#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

#     img = torch.from_numpy(img).to(device)
#     img = img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
#     return img

# def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
#     # Rescale coords (xyxy) from img1_shape to img0_shape
#     if ratio_pad is None:  # calculate from img0_shape
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
#     coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
#     coords[:, :10] /= gain
#     coords[:, 0].clamp_(0, img0_shape[1])  # x1
#     coords[:, 1].clamp_(0, img0_shape[0])  # y1
#     coords[:, 2].clamp_(0, img0_shape[1])  # x2
#     coords[:, 3].clamp_(0, img0_shape[0])  # y2
#     coords[:, 4].clamp_(0, img0_shape[1])  # x3
#     coords[:, 5].clamp_(0, img0_shape[0])  # y3
#     coords[:, 6].clamp_(0, img0_shape[1])  # x4
#     coords[:, 7].clamp_(0, img0_shape[0])  # y4
#     coords[:, 8].clamp_(0, img0_shape[1])  # x5
#     coords[:, 9].clamp_(0, img0_shape[0])  # y5
#     return coords

# def get_face(input_image):
#     # Parameters
#     size_convert = 128
#     conf_thres = 0.4
#     iou_thres = 0.5
    
#     # Resize image
#     img = resize_image(input_image.copy(), size_convert)

#     # Via yolov5-face
#     with torch.no_grad():
#         pred = model(img[None, :])[0]

#     # Apply NMS
#     det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
#     bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    
#     landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
#     return bboxs, landmarks

# def main():
#     # Open camera 
#     cap = cv2.VideoCapture(0)
#     start = time.time_ns()
#     frame_count = 0
#     fps = -1
    
#     # Save video
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
    
#     size = (frame_width, frame_height)
#     video = cv2.VideoWriter('results/face-detection.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
#     # Initialize SORT tracker
#     from sort import Sort
#     sort_tracker = Sort()
#     while(True):
#         _, frame = cap.read()
#         # Get faces
#         bboxs, landmarks = get_face(frame)

#         # Apply SORT tracker
#         dets_to_sort = np.empty((0, 5))

#         if bboxs is not None:
#             for bbox in bboxs:
#                 x1, y1, x2, y2 = bbox
#                 dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, 0])))  # 0 is the default class ID

#             # Run SORT tracker
#             tracks = sort_tracker.update(dets_to_sort)
#             #tracks = sort_tracker.getTracks()

#             # Visualize tracks on the image
#             for track in tracks:
#                 track_id = int(track[-1])  # Extract track ID from the last element
#                 color = (int(track_id * 255 / len(tracks)) % 255, 255, 255)  # Assign a color based on track ID

#                 # Draw bounding box
#                 cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 2)

#                 # Draw track ID
#                 cv2.putText(frame, str(track_id), (int(track[0]), int(track[1]) - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         #Show result
#         cv2.imshow("Face Detection", frame)
        
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break  
    
#     video.release()
#     cap.release()
#     cv2.destroyAllWindows()
#     cv2.waitKey(0)

# if __name__ == "__main__":
#     main()
