#!/usr/bin/env python3
from feat import Detector
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "rf"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model=landmark_model,
                    au_model = au_model, emotion_model=emotion_model)