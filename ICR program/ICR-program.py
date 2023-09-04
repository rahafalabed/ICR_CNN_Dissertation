import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageEnhance

# Choose model
model = load_model("Final models/my_best_model3.h5")

class_indices = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '.', 13: ','
}

def show_info_hover(event, x, y, flags, param):
    global current_image
    
    if event == cv2.EVENT_MOUSEMOVE:
        temp_image = current_image.copy()
        
        for i, cnt in enumerate(contours):
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                symbol = symbol_info[i][0]
                confidence = symbol_info[i][1]
                x, y, w, h = cv2.boundingRect(cnt)
                
                color = (0, 150, 0)  # Default color is green
        
                if confidence < 60:
                    color = (0, 0, 255)  # Change color to red

                # Draw the bounding box with the determined color
                cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 1)
                
                # Draw the text with the determined color
                data = f"{symbol}, {confidence}%"
                cv2.putText(temp_image, data, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for i, cnt in enumerate(contours):
            symbol = symbol_info[i][0]
            confidence = symbol_info[i][1]
            
            if confidence < 60:  # Only show with confidence < 60
                x, y, w, h = cv2.boundingRect(cnt)
                color = (0, 0, 255)  # Change color to red
                cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 1)
                # Draw the text with the color
                data = f"{symbol}, {confidence}%"
                cv2.putText(temp_image, data, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        cv2.imshow('Image output', temp_image)

def show_all():
    global current_image
    
    if current_image is None:
        return
    
    temp_image = current_image.copy()
    
    for i, cnt in enumerate(contours):
        symbol = symbol_info[i][0]
        confidence = symbol_info[i][1]
        x, y, w, h = cv2.boundingRect(cnt)

        color = (0, 150, 0)  # Default color is green
        
        if confidence < 60:
            color = (0, 0, 255)  # Change color to red
        
        # Draw the bounding box with the determined color
        cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 1)
        
        # Draw the text with the determined color
        data = f"{symbol}, {confidence}%"
        cv2.putText(temp_image, data, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('Image output', temp_image)
    cv2.setMouseCallback('Image output', lambda *args: None)  # Disable mouse callback

def enable_hover():
    cv2.setMouseCallback('Image output', show_info_hover)

def show_low_confidence():
    global current_image
    
    if current_image is None:
        return
    
    temp_image = current_image.copy()
    
    for i, cnt in enumerate(contours):
        symbol = symbol_info[i][0]
        confidence = symbol_info[i][1]
        
        if confidence < 60:  # Only show with confidence < 60
            x, y, w, h = cv2.boundingRect(cnt)
            color = (0, 0, 255)  # Change color to red
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 1)
            # Draw the text with the color
            data = f"{symbol}, {confidence}%"
            cv2.putText(temp_image, data, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('Image output', temp_image)
    cv2.setMouseCallback('Image output', lambda *args: None)  # Disable mouse callback

# Function to enhance the image (increase contrast and brightness)
def enhance_image(sample_image_array):
    enhancer = ImageEnhance.Contrast(Image.fromarray(sample_image_array))
    enhanced_image_array = np.array(enhancer.enhance(2.0))  # Increase contrast 
    return enhanced_image_array

def Recognition(image_path):
    global current_image, symbol_info
    
    symbol_info = []
    current_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if current_image is None:
        print("Error loading the image.")
        return
    
    # Enhance the image using the enhancement function
    current_image = enhance_image(current_image)
    
    make_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(make_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    global contours
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if not contours:
        print("Nothing detected.")
        return
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Add padding to the bounding box
        padding = 10  # Adjust this value as needed
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        
        # Ensure the ROI coordinates are within the image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, current_image.shape[1] - x)
        h = min(h, current_image.shape[0] - y)
        roi = th[y:y+h, x:x+w]

        try:
            img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except cv2.error:
            print("Error resizing.")
            continue
        
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        prediction = model.predict([img])[0]
        final = np.argmax(prediction)
        confidence = int(max(prediction) * 100)
        symbol = class_indices.get(final, '?')
        
        symbol_info.append((symbol, confidence))

    create_buttons()  # Create buttons immediately after loading the image
    show_all()  # Show all by default  

def open_image():
    global current_image
    current_image = None  # Clear the current image before loading a new one
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.pdf")])
    if file_path:
        Recognition(file_path)

def create_buttons():
    btn_show_all = Button(button_frame, text="Show All", fg='black', command=show_all)
    btn_show_all.grid(row=0, column=0, pady=10)
    
    btn_hover = Button(button_frame, text="Enable Hover", fg='black', command=enable_hover)
    btn_hover.grid(row=1, column=0, pady=10)

    btn_low_confidence = Button(button_frame, text="Show Low Confidence", fg='black', command=show_low_confidence)
    btn_low_confidence.grid(row=2, column=0, pady=10)

source = Tk()  # Create the main window
source.title("ICR System")

button_frame = Frame(source)
button_frame.grid(row=1, column=0, pady=10)

btn_load = Button(source, text="Load Image for ICR", fg='black', command=open_image)
btn_load.grid(row=0, column=0, pady=10)

source.mainloop()