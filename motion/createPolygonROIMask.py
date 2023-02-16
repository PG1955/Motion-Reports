import sys

import cv2
import numpy as np
import argparse

# initializing an argument parser object
ap = argparse.ArgumentParser()

# Allow cmd line input of image and mask file names.
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-m", "--mask", required=True, help="Path for writing the mask")
ap.add_argument("-z", "--zoom", help="Zoom factor integer", default=1)

# parsing the argument
args = vars(ap.parse_args())
zoom = int(args["zoom"])

print(f'Zoom level is set to {zoom}')
print('Mark the points you require to form a ROI.')
print('Hit the c key to Complete the ROI')
print('Keep adding areas.')
print('When complete hit the p key to Preview the mask.')
print('To write a new mask hit the s key to Save it.')
print('Hit escape key to abort.')

# Create log_point matrix get coordinates of mouse click on image
array = np.empty((0, 3), np.int32)

counter = 0
circle = 1


def mousePoints(event, x, y, flags, params):
    global counter
    global array
    global circle
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        array = np.append(array, np.array([[circle, x, y]]), axis=0)
        counter = counter + 1


def getPoints(array, index):
    # print(f'Inbound array shape is {array.shape}')
    points = np.empty((0, 2), np.int32)
    for x, y, z in array:
        if x == index:
            points = np.append(points, np.array([[y, z]]), axis=0)
    return points


image = cv2.imread(args["image"])
image_height = image.shape[0]
image_width = image.shape[1]

print(f'Width:{image_width} Height:{image_height}')

if zoom > 1:
    image_width *= zoom
    image_height *= zoom
    image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
# Create blank mask the same size as the image.
mask = np.zeros(image.shape[:2], dtype="uint8")
print(f'Mask shape is {mask.shape} and mask dtype is {mask.dtype}')

while True:

    if counter > 0:
        points = getPoints(array, circle)
        # print(f'Shape of points is {points.shape}')
        cv2.circle(image, (points[counter - 1]), 3, (0, 255, 0), cv2.FILLED)

    key = cv2.waitKey(3)
    # if key == 27:
    if key == ord('c'):
        color = (100, 100, 100)
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.imshow("Original Image ", image)
        cv2.fillPoly(mask, [points], color=(255, 0, 0))
        circle += 1
        counter = 0

    if key == ord('p'):
        cv2.imshow("Mask", mask)
        break

    if key == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)

    # Showing original image
    cv2.imshow("Original Image ", image)
    # Mouse click event on original image
    cv2.setMouseCallback("Original Image ", mousePoints)

cv2.waitKey(0)
if zoom > 1:
    image_height = int(image_height / zoom)
    image_width = int(image_width / zoom)
    mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

if not cv2.imwrite(args["mask"], mask):
    raise Exception(f'Could not write {args["mask"]}')
else:
    print(f'Written file to {args["mask"]}')

cv2.destroyAllWindows()
sys.exit(0)
