import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pygame
from sys import exit
import random

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Goal Blocker')
clock = pygame.time.Clock()
test_font = pygame.font.Font('/Users/gurno/Downloads/Font.ttf', 40)
font_surface = test_font.render('Press Q to Quit and Restart.', False, (0, 255, 0))
soccerball = pygame.image.load('/Users/gurno/Downloads/soccer_ball.png')
soccerball_rect = soccerball.get_rect(center = (320, 240))
gloves = pygame.image.load('/Users/gurno/Downloads/newgloves (1).png')
soccer_stadium = pygame.image.load('/Users/gurno/Downloads/soccer-stadium (1).png')
soccer_stadium_rect = soccer_stadium.get_rect()

soccerballrand_x = random.randint(-5, 5) * 3
soccerballrand_y = random.randint(-5, 5) * 3

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()

        w, h, c = frame.shape

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        coordinate_landmarks = results.multi_hand_landmarks

        if coordinate_landmarks:
            for handLMs in coordinate_landmarks:
                x_max = 0
                y_max = 0
                x_min = h
                y_min = w
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(frame, (w - x_min, y_min), (w - x_max, y_max), (255, 255, 255), 2)

                gloves_rect = (x_min, y_min, x_max, y_max)

                soccerball_rect.x = soccerball_rect.x + soccerballrand_x
                soccerball_rect.y = soccerball_rect.y + soccerballrand_y

                screen.blit(soccer_stadium, soccer_stadium_rect)
                screen.blit(gloves, gloves_rect)
                screen.blit(soccerball, soccerball_rect)

                if pygame.Rect.colliderect(soccerball_rect, gloves_rect):
                    screen.fill((255, 0, 0))
                    screen.blit(font_surface, (160, 240))
                
                pygame.display.update()
                clock.tick(60)
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
            
        
        cv2.imshow('Hand Tracking', image)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()