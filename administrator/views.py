from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views import View

from .form import RegForm, crimeform, fireform
from .models import CriminalTable, FireTable, RegisterTable

# Create your views here.

class add_missing(View):
    def get(self, request):
        return render(request, "add missing.html")

class registration(View):
    def get(self,request):
        return render(request,"registration.html")
    def post(self, request):
        form = RegForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponse('''<script>alert("registerd successfully"); window.location="/registration"</script>''')

           
class add_new_criminal(View):
    def get(self,request):
        return render(request,"add new criminal.html")  
    def post(self,request):
        form=crimeform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse('''<script>alert("registered successfully"); window.location="/view_criminal"</script>''')


class Edit_criminal(View):
    def get(self,request,id):
        c=CriminalTable.objects.get(id=id)
        return render(request,"editcriminal.html",{'val':c})
    def post(self,request,id):
        c=CriminalTable.objects.get(id=id)
        form=crimeform(request.POST,request.FILES,instance=c)
        if form.is_valid():
            form.save()
            return redirect('view_criminal')
class complaint(View):
    def get(self,request):
        return render(request,"complaint.html") 
    

class view_criminal(View):
    def get(self,request):
        obj = CriminalTable.objects.all()
        return render(request,"criminal.html", {'val': obj})
 
class delete_c(View):
    def get(self, request, c_id):
        obj =  CriminalTable.objects.get(id=c_id)
        obj.delete()
        return HttpResponse('''<script>alert("delete successfully"); window.location="/view_criminal"</script>''')


class view_user(View):
    def get(self,request):
        obj =RegisterTable.objects.all()
        return render(request,"view users.html", {'val': obj})
    
class delete_u(View):
    def get(self, request, u_id):
        obj =RegisterTable.objects.get(id=u_id)
        obj.delete()
        return HttpResponse('''<script>alert("delete successfully"); window.location="/view_user"</script>''')
    
class add_fire(View):
    def get(self,request):
        return render(request,"addfire.html")  
    def post(self,request):
        form=fireform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse('''<script>alert("registered successfully"); window.location="/view_fire"</script>''')


class Edit_fire(View):
    def get(self,request,id):
        c=FireTable.objects.get(id=id)
        return render(request,"editfire.html",{'val':c})
    def post(self,request,id):
        f=FireTable.objects.get(id=id)
        form=fireform(request.POST,request.FILES,instance=f)
        if form.is_valid():
            form.save()
            return redirect('view_fire')
        
class view_fire(View):
    def get(self,request):
        obj = FireTable.objects.all()
        return render(request,"fire.html", {'val': obj})
    
class delete_f(View):
    def get(self, request, f_id):
        obj =FireTable.objects.get(id=f_id)
        obj.delete()
        return HttpResponse('''<script>alert("delete successfully"); window.location="/view_fire"</script>''')
    


# import os
# import cv2
# import face_recognition
# import numpy as np
# import telepot
# from datetime import datetime
# from django.views import View
# from .models import Camera, Alert, CriminalTable, RegisterTable # Adjust based on actual models

# # Telegram bot setup
# TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
# bot = telepot.Bot(TELEGRAM_BOT_TOKEN)

# # Load YOLO model
# yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# layer_names = yolo_net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
# with open("coco.names", "r") as f:
#     yolo_classes = [line.strip() for line in f.readlines()]

# class Monitor_camera(View):
#     def get(self, request, id):
#         # Path to the known images folder and the text file with names
#         known_images_path = "C:/Users/user/Desktop/djangoprojects/patrolwatchfile/projectpatrolwatch/media/criminalimages/known_images"
#         names_file = "C:/Users/user/Desktop/djangoprojects/patrolwatchfile/projectpatrolwatch/media/known_images/known_faces.txt"

#         # Function to load names from the text file
#         def load_names(names_file):
#             with open(names_file, "r") as f:
#                 names = [line.strip() for line in f.readlines()]
#             return names

#         # Load known face names from the text file
#         person_names = load_names(names_file)
#         known_face_encodings = []
#         known_face_names = []

#         # Load known faces
#         for i, person_folder in enumerate(os.listdir(known_images_path)):
#             person_folder_path = os.path.join(known_images_path, person_folder)
#             person_name = person_names[i]

#             for image_file in os.listdir(person_folder_path):
#                 image_path = os.path.join(person_folder_path, image_file)
#                 image = face_recognition.load_image_file(image_path)
#                 face_encoding = face_recognition.face_encodings(image)
#                 if face_encoding:
#                     known_face_encodings.append(face_encoding[0])
#                     known_face_names.append(person_name)
#                     print(f"Encoded {image_file} for {person_name}")
#                 else:
#                     print(f"No face found in {image_file}, skipping")

#         try:
#             camera = Camera.objects.get(id=id)
#             CHAT_ID = camera.fire_station.chat_id if camera.fire_station else camera.police_station.chat_id

#             cap = cv2.VideoCapture(0)  # Use camera ID if multiple cameras
#             recording = False
#             video_writer = None

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Fire detection
#                 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#                 mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
#                 contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                 for cnt in contours:
#                     if cv2.contourArea(cnt) > 5000:
#                         Alert.objects.create(alert_type='FIRE', camera=camera, details="Fire detected")
#                         bot.sendMessage(CHAT_ID, "Fire detected in the monitored area.")
#                         break

#                 # Face recognition
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 face_encodings = face_recognition.face_encodings(rgb_frame)
#                 for face_encoding in face_encodings:
#                     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                     if True in matches:
#                         match_index = matches.index(True)
#                         criminal_name = known_face_names[match_index]
#                         Alert.objects.create(alert_type='INTRUSION', camera=camera, details=f"Intrusion by {criminal_name}")
#                         bot.sendMessage(CHAT_ID, f"Intrusion detected by {criminal_name}.")
#                         break

#                 # Person detection using YOLO
#                 blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#                 yolo_net.setInput(blob)
#                 detections = yolo_net.forward(output_layers)
#                 person_detected = False

#                 for detection in detections:
#                     scores = detection[5:]
#                     class_id = np.argmax(scores)
#                     confidence = scores[class_id]
#                     if confidence > 0.5 and yolo_classes[class_id] == "person":
#                         person_detected = True
#                         x, y, w, h = map(int, detection[0:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                 # Video recording
#                 if person_detected and not recording:
#                     recording = True
#                     video_file_path = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
#                     video_writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))
#                 elif not person_detected and recording:
#                     recording = False
#                     video_writer.release()
#                     with open(video_file_path, 'rb') as video_file:
#                         bot.sendVideo(CHAT_ID, video_file)

#                 if recording:
#                     video_writer.write(frame)

#                 cv2.imshow("Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             cap.release()
#             if recording:
#                 video_writer.release()
#             cv2.destroyAllWindows()
#         except Exception as e:
#             print(f"Error in camera monitoring: {e}")



import cv2
import face_recognition
from django.http import StreamingHttpResponse
from .models import CriminalTable
import numpy as np

# Load known criminals
def load_criminals():
    criminals = CriminalTable.objects.all()
    known_face_encodings = []
    known_face_names = []

    for criminal in criminals:
        image = face_recognition.load_image_file(criminal.Image.path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(criminal.Criminalname)
    
    return known_face_encodings, known_face_names

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # 0 for laptop webcam

    def __del__(self):
        self.video.release()

    def get_frame(self, known_face_encodings, known_face_names):
        success, frame = self.video.read()
        if not success:
            return None

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw box around face
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera, known_face_encodings, known_face_names):
    while True:
        frame = camera.get_frame(known_face_encodings, known_face_names)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    known_face_encodings, known_face_names = load_criminals()
    return StreamingHttpResponse(gen(VideoCamera(), known_face_encodings, known_face_names),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'index.html')