import cv2
import matplotlib.pyplot as plt
import numpy as np

print("package imported")

cap = cv2.VideoCapture("Resources/input.mp4")
car_cascade = cv2.CascadeClassifier("Resources/cars.xml")

def do_canny(img):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, 50, 150)
    return canny

def displayLines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1,y1), (x2,y2), (255, 0, 0), 5)
    return line_img

def ROI(img):
    height = img.shape[0]   #gets y value of bottom
    polygons = np.array([
        [(100, height), (800, height), (380,290)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)   #operation to only shows ROI
    return masked_image

def carDetection(CD_img):
    gray = cv2.cvtColor(CD_img, cv2.COLOR_RGB2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)   #Detects cars of different sizes
    for (x,y,w,h) in cars:
        cv2.rectangle(CD_img, (x,y),(x+h,y+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(CD_img, 'Car', (x+6,y-6), font, 0.5, (0, 0, 255),1)
    return CD_img

while (cap.isOpened()):
    success, img = cap.read()
    lane_image = np.copy(img)
    CDImage = np.copy(img)  # img for car detection
    canny = do_canny(img)
    cropped_image = ROI(canny)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_img = displayLines(lane_image, lines)
    combined_vid = cv2.addWeighted(lane_image, 0.8, line_img, 1, 1)
    carDetectionImage = carDetection(CDImage)

    combined_vid_detection = cv2.addWeighted(combined_vid, 0.8, carDetectionImage, 1, 1)

    cv2.imshow("Video", combined_vid_detection)
    if cv2.waitKey(1) & 0xFF == ord('q'): #adds a delay and looks for the letter q to break video
        break

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()
