import cv2
import numpy as np
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)
video_capture = cv2.VideoCapture(0)

# variables to handle video recording
out = None
recording = False
frames_recorded = 0  # Initialize frames_recorded to 0
max_frames_to_record = 180  

def motion_detection():
    global video_capture, recording, out, frames_recorded

    prev_frame = None
    curr_frame = None
    # Initialize the recording counter
    recording_count = 1  
    

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        if curr_frame is None:
            curr_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(curr_frame, gray_frame)
        thresh_frame = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # contours of moving objects
        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue

            # bounding box around the detected motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # video record if not already recording
            if not recording:
                recording = True
                recording_count += 1  # Increment recording count
                timestamp = int(time.time())  # Use timestamp as part of the filename
                video_filename = f'motion_detected_video_{recording_count}_{timestamp}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

                # Notify the frontend via WebSocket
                socketio.emit('motion_detected')

                print("Motion detected!")

        if recording:
            # Write the frame to the video file
            out.write(frame)
            frames_recorded += 1

            if frames_recorded >= max_frames_to_record:
                # Stop recording after reaching the maximum frames
                recording = False
                frames_recorded = 0
                out.release()

        prev_frame = curr_frame
        curr_frame = gray_frame

        # frame as JPEG to display in the Flask web application
        _, jpeg_frame = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(motion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
