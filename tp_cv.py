import cv2
from ultralytics import YOLO

model = YOLO(r"D:\projects\runs\detect\train5\weights\best.pt")
print("model loaded")

cam=cv2.VideoCapture(0)

if not cam.isOpened():
    print("cam not open")
    exit()

while True:
    ret,frame=cam.read()
    if not ret:
        print("Fail to grab frame")
        break

    results = model(frame)
    annoted_frame=results[0].plot()
    # print(results)
    cv2.imshow("Knife Detected",annoted_frame)

    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()