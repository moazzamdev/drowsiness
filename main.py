import torch
import numpy as np
import cv2
import posixpath
import pathlib
import time
import pygame
import sys
import importlib.util
import ultralytics

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', path="model/train/exp/weights/last.pt", )
model.conf = 0.55
model.iou = 0.4
model.classes = [4, 5]
lst = []
drowsiness_count = 0
cap = cv2.VideoCapture(0)

def countdown_timer(seconds):
    while seconds:
        mins, secs = divmod(seconds, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        seconds -= 1


# Initialize Pygame mixer
pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load("model/alarm_clock_old-[AudioTrimmer.com].mp3")
while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame, size=320)

    # Print the results_render for inspection
    df = results.pandas().xyxy[0]

    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    for label in df['name']:
        if label == 'drowsiness':
            drowsiness_count += 1
            #print("Drowsiness count:", drowsiness_count)

            if drowsiness_count > 15:
                print("Alarm activated!")
                pygame.mixer.music.play()

                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                countdown_timer(3)
                break  # Stop looping if count exceeds 150
        elif label == 'awake':
            drowsiness_count = 0  # Reset count if 'awake' is detected

    # If the loop completes without breaking, print the final count
    if drowsiness_count <= 15:
        print("Final drowsiness count:", drowsiness_count)


cap.release()
cv2.destroyAllWindows()