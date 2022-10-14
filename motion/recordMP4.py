import cv2
import time
import datetime

def write_timestamp(wt_frame):
    # Write data and time on the video.
    wt_now = datetime.datetime.now()
    wt_text = wt_now.strftime("%Y/%m/%d %H:%M:%S")
    #wt_font = cv2.FONT_HERSHEY_PLAIN
    wt_font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.75
    thickness = 2
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(wt_text, wt_font, font_scale, thickness)
    line_height = text_size[1] + 5
    cv2.putText(wt_frame, wt_text,
                (360, 460),
                wt_font,
                font_scale,
                (255, 255, 255),
                thickness, line_type)
    return wt_frame

# The duration in seconds of the video captured
capture_duration = 10

cap = cv2.VideoCapture(0)
width = 640
height = 480

cap.set(3,width)
cap.set(4,height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Motion/output.mp4', fourcc, 20.0, (width, height))


start_time = time.time()
while( int(time.time() - start_time) < capture_duration ):
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # Add timestamp.
    frame = write_timestamp(frame)
    out.write(frame)
    # cv2.imshow('frame', frame)
    c = cv2.waitKey(5)
    if c & 0xFF == ord('q'):
        break
        
out.release()
cap.release()
out.release()
cv2.destroyAllWindows()

