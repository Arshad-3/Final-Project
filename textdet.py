import cv2
import google.generativeai as genai
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3

cap = cv2.VideoCapture(0)
genai.configure(api_key="AIzaSyBpkteNGTau02yx-Nfu9WkLy-P0JNXwpDU")
# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
ret,processed = cap.read()

#image to grayscale
# processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold = 0.25

text_ = reader.readtext(processed)

# daw bbox and text
text_list = []
for t_, t in enumerate(text_):
    
    bbox, text, score = t
    if score > threshold:
        print(t)
        # Extract bounding box points
        (x_min, y_min) = bbox[0]  # Top-left corner
        (x_max, y_max) = bbox[2]  # Bottom-right corner
        # Draw rectangle
        cv2.rectangle(processed, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # Put text
        cv2.putText(processed, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
        text_list.append(text)

speech = " ".join(text_list)
print(speech)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(processed)
plt.axis("off")
plt.show()

prompt = f"Rewrite the following text into a grammatically correct and meaningful sentence.: {speech}"
model = genai.GenerativeModel("gemini-1.5-pro")  # Use Gemini Pro model
response = model.generate_content(prompt)
meaningful_sentence = response.text.strip()
print("Corrected Sentence:", meaningful_sentence)

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 0 = Male, 1 = Female
engine.setProperty('rate', 125)

engine.say(meaningful_sentence)

# Play the speech
engine.runAndWait()