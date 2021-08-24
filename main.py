from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
from imutils.video import FPS
import numpy as np
import imutils
import argparse
import cv2
import time
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="Path to trained model")
ap.add_argument("-l", "--labels", required=True,
                help="Path to model's labels")
args = vars(ap.parse_args())

print("[INFO] Loading model...")
model = load_model(args["model"])
labels = pickle.loads(open(args["labels"], 'rb').read())

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2.0)
fps = FPS().start()

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3))
    edged = cv2.Canny(blurred, 165, 80)

    # Finding and sorting contours left to right, error handling prevents
    # script's crashes in case of no contours were detected
    try:
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        contours = sort_contours(contours, method="left-to-right")[0]
    except ValueError:
        pass

    chars = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        # Filtering out bounding boxes by their sizes
        if (5 <= w <= 150) and (15 <= h <= 120):
            # Extracting image with the character and applying thresholding
            # on it
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # Resizing by bigger dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

            (tH, tW) = thresh.shape

            # If NN needs to be provided with a RGB input image
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # Determining padding, so 32x32 shape is preserved
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # Padding the image
            padded = cv2.copyMakeBorder(thresh,
                                        top=dY, bottom=dY,
                                        left=dX, right=dX,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # Preparing padded image for classification
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # Defining initial predictions shape in case of no characters were found
    # to predict (100 dimension does not matter, it could be anything, only 36
    # is important, which is equal to the number of classes)
    preds_shape = (100, 36)

    # Trying to make a prediction, if there are no characters, predictions array
    # is full of zeros and has shape (100, 36)
    try:
        preds = model.predict(chars)
    except UnboundLocalError:
        preds = np.zeros(shape=preds_shape)

    label_names = list(labels.keys())

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)

        prob = pred[i]
        label = label_names[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {prob * 100:.2f}%", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    fps.update()
    time.sleep(0.05)

fps.stop()
print(f"[INFO] Elapsed time in seconds: {fps.elapsed():.2f}")
print(f"[INFO] Approx. FPS: {fps.fps():.2f}")

cv2.destroyAllWindows()
