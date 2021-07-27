import cv2
import PIL.Image as Image
import io
import time
import jinja2
from datetime import date
from datetime import datetime
import os
from bson.objectid import ObjectId
import uuid
import shutil
import pymongo
import gridfs
import numpy as np
import random
from flask_cors import CORS
from dotenv import load_dotenv
from utils.sendSMS import send_bulk_sms
from flask import (
    Flask,
    jsonify,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for, Response
)
import base64
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
from twilio.rest import Client
import threading
#from camera import VideoCam
# Violence Libraries etc

from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from model import prediction

client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
mydb = client['ojo']
fs = gridfs.GridFS(mydb)
userInfo = mydb.userInfo
logs = mydb.logStorage
camInfo = mydb.camInfo
staffLogInfo = mydb.staffLogInfo
detectionInfo = mydb.detectionInfo
username1 = ""
password1 = ""
detectionState = "false"
value = "1"


class Userlogs:
    def add_login_time(self, username, name, date, time):
        login_entry = {
            "_id": uuid.uuid4().hex,
            "username": username,
            "name": name,
            "date": date,
            "time": time,
            "logged_in": "true"

        }
        staffLogInfo.insert_one(login_entry)
        return jsonify(login_entry), 200

    def add_logout_time(self, username, name, date, time):
        logout_entry = {
            "_id": uuid.uuid4().hex,
            "username": username,
            "name": name,
            "date": date,
            "time": time,
            "logged_in": "false"

        }
        staffLogInfo.insert_one(logout_entry)
        return jsonify(logout_entry), 200


class Detection:
    def setDetectionState(self, query, detectionState):
        update = {

            "$set": {
                "detection": detectionState
            }

        }
        queryName = {
            '_id': query
        }

        detectionInfo.update_one(queryName, update)

        pass

    def getDetectionState(self, id):
        return detectionInfo.find_one({'_id': id})


class Camera:
    def addCamera(self, request):
        cam = {
            "_id": uuid.uuid4().hex,
            "ip": request.form.get('ip1'),
            "enabled": "true"
        }

        camInfo.insert_one(cam)
        return True

    def removeCamera(self, request):
        cam = {

            "_id": uuid.uuid4().hex,
            "ip": request.form.get('ip1'),
            "enabled": "false"

        }
        camInfo.remove(cam)
        return jsonify(cam), 200

    def updateCamera(self, query, camStatus):
        camUpdate = {

            "$set": {
                "enabled": camStatus
            }

        }

        camInfo.update_one(query, camUpdate)

        return jsonify(camUpdate), 200

    def updateCameraIP(self, ip):
        ipUpdate = {

            "$set": {
                "ip": ip
            }
        }
        query = {
            "_id": "7ee2efadbd4f463abea9214d07a73fc2"
        }

        camInfo.update_one(query, ipUpdate)
        return jsonify(ipUpdate), 200

    def getCameraIP(self):
        ipVal = ""
        cursor = camInfo.find({})
        for x in cursor:
            if (x['_id'] == "7ee2efadbd4f463abea9214d07a73fc2"):
                ipVal = x['ip']

        return ipVal


class Member:

    def start_session(self, member):
        session['logged_in'] = True

        return jsonify(member), 200

    def signIn(self, username, password):
        member = {
            "username": username,
            "password": password,
        }

        if (userInfo.find_one(member)):
            return True
        else:
            return False

    def addMember(self, request):
        member = {
            "_id": uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "username": request.form.get('username'),
            "password": request.form.get('password'),
            "phone": request.form.get('phone'),
            "role": request.form.get('role')
        }
        userInfo.insert_one(member)

        return jsonify(member), 200

    def removeMember(self, request):
        member = {

            "email": request.form.get('email'),
            "username": request.form.get('username')

        }
        userInfo.remove(member)

        return jsonify(member), 200

    def updateInfo(self, query, request):
        member = {

            "$set": {
                "name": request.form.get('name'),
                "email": request.form.get('email'),
                "username": request.form.get('username'),
                "phone": request.form.get('phone'),
                "password": request.form.get('password')
            }

        }

        userInfo.update_one(query, member)

        return jsonify(member), 200

    def updateRole(self, userName, rolename):
        role = {
            "$set": {
                "role": rolename
            }
        }
        query = {
            "username": userName
        }

        userInfo.update_one(query, role)

        return jsonify(role), 200


app = Flask(__name__)
app.secret_key = 'somesecretkeythatonlyishouldknow'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app.config['SERVER_NAME'] = "127.0.0.1:5000"
app.jinja_env.globals.update(zip=zip)


def get_file_object(file_id):


    out = fs.get(ObjectId(file_id)).read()
    return (out)



@app.before_request
def before_request():
    g.user = None
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['_id'] == session['user_id']):
                user = x
                g.user = user


@app.route('/login', methods=['GET', 'POST'])
def login():
    valdity = False
    session.pop('user_id', None)
    if request.method == 'POST':
        session.pop('user_id', None)

        username1 = request.form['username']
        password1 = request.form['password']

        userData = userInfo.find_one({'username': username1})
        if (userData):
            if (userData['password'] == password1):
                session['user_id'] = userData['_id']
                # Add Login Time and Date
                user_log = Userlogs()
                today = date.today()
                today1 = today.strftime("%B %d, %Y")
                nows = datetime.now()
                current_time = nows.strftime("%H:%M:%S")

                user_log.add_login_time(userData['username'], userData['name'], today1, current_time)

                return redirect(url_for('dashboard'))
            valdity = True
            return render_template('login.html', valdity=valdity)
        valdity = True
        return render_template('login.html', valdity=valdity)

    return render_template('login.html', valdity=valdity)


@app.route('/byteArrayDisplay')
def byteArrayDisplay():
    cursor = logs.find({})
    val = ""

    detectImageList = []
    for x in cursor:
        val = x['imageId']
        d = get_file_object(val)
        str_equivalent_image = base64.b64encode(io.BytesIO(d).getvalue()).decode()
        img_tag = "<img width='100' height = '100' src='data:image/png;base64," + str_equivalent_image + "'/>"
        detectImageList.append(img_tag)

    return Response(io.BytesIO(d).getvalue(), mimetype='image/png')


@app.route('/logout')
def logout():
    # Update Login Date
    # Update Login time
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    user_log = Userlogs()
    today = date.today()
    today1 = today.strftime("%B %d, %Y")
    nows = datetime.now()
    current_time = nows.strftime("%H:%M:%S")

    user_log.add_logout_time(user['username'], user['name'], today1, current_time)

    session.pop('logged_in', None)
    return redirect(url_for('login'))


def gen(val=0):
    detectionState = "false"
    det = Detection()
    id = "9de1c199f1764b2b9afa12d2b5eeb8bb"
    det.setDetectionState(id, detectionState)
    cap = cv2.VideoCapture(val)

    model = cv2.dnn.readNet('yolov3_training_final.weights',
                            'yolov3_training.cfg')

    classes = []
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()
    i = 0
    frame_rate = 30
    prev = 0
    detectIterator = 0

    while True:
        ret, image = cap.read()
        img = image
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            img = image
            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            output_layer_names = model.getUnconnectedOutLayersNames()
            ########Forward Pass################
            layer_output = model.forward(output_layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_output:
                for prediction in output:
                    scores = prediction[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if (confidence > 0.4):
                        center_x = int(prediction[0] * width)
                        center_y = int(prediction[1] * height)
                        w = int(prediction[2] * width)
                        h = int(prediction[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
            colors = np.random.uniform(0, 255, size=(len(indexes), 3))
            if (len(indexes) != 0):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (label != "Human_face"):
                        confidence = str(round(confidences[i], 2))
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        suf = random.randint(0, 100)
                        filenamefull = "file" + str(suf)

                        ret, jpeg = cv2.imencode('.jpg', img)
                        data = []
                        data.append(jpeg.tobytes())
                        frame1 = data[0]
                        today = date.today()
                        today1 = today.strftime("%B %d, %Y")
                        nows = datetime.now()
                        current_time = nows.strftime("%H:%M:%S")

                        fs.put(frame1, filename=filenamefull)
                        imgId = str(fs.put(frame1, filename=filenamefull))
                        insertDataLog(current_time, today1, imgId, "Weapon","Live Stream")



                        img[:, :, 0] = 0
                        img[:, :, 1] = 0
                        detectionState = "true"

                        today = date.today()
                        today1 = today.strftime("%B %d, %Y")
                        nows = datetime.now()
                        current_time = nows.strftime("%H:%M:%S")


                        detectIterator += 1

                    # cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            image = img

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        frame = data[0]
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if (detectionState == "true" and detectIterator == 1):
            detectIterator = 0
            # det = Detection()
            id = "9de1c199f1764b2b9afa12d2b5eeb8bb"
            det.setDetectionState(id, detectionState)

            # Now store/put the image via GridFs object.


            #

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            break;

    # cap.release()
    # cv2.destroyWindow()


value = "1"


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    c1 = Camera()
    if request.method == 'POST':
        print(request.form['ip1'])
        cursor1 = camInfo.find({})
        for x in cursor1:
            if (x['name'] == "webcam"):
                cam = x
        quer = {
            "name": "webcam"
        }

        enable = "true"
        rep = c1.updateCamera(quer, enable)
        d1 = Detection()
        rep = d1.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")
        c1.updateCameraIP(request.form['ip1'])
        return rep

    val = c1.getCameraIP()
    if (val == "0"):
        val = int(val)

    if (val == "1"):
        val = int(val)

    return Response(gen(val), mimetype='multipart/x-mixed-replace; boundary=frame')


def insertDataLog(time, date, imageId,type,mode):
    t = {
        "time": time,
        "date": date,
        "imageId": imageId,
        "detectionType":type,
        "videoType": mode
    }
    logs.insert_one(t)


@app.route('/log', methods=['GET', 'POST'])
def log():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})

    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    logs = mydb.logStorage
    abc = logs.find()
    cursor = logs.find({})


    val = ""
    detectImageList = []



    for x in cursor:
        val = x['imageId']
        d = get_file_object(val)
        str_equivalent_image = base64.b64encode(io.BytesIO(d).getvalue()).decode()

        img_tag = "data:image/png;base64," + str_equivalent_image

        detectImageList.append(img_tag)


    return render_template('log.html', logs=abc, user=user, detectImageList=detectImageList)


@app.route('/')
def main():
    return redirect(url_for('dashboard'))


@app.route('/test')
def test():
    return render_template("test.html")


@app.route('/evaluate')
def evaluate():
    if not g.user:
        return redirect(url_for('login'))
    target = os.path.join(APP_ROOT, "static/detectVideo/")
    target1 = os.path.join(APP_ROOT, "static/detectImage/")

    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)

    if os.path.exists(target1):
        shutil.rmtree(target1)
    os.makedirs(target1)

    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if x['username'] == g.user['username']:
                user = x
    # if request.method == 'POST':
    #    render_template('evaluate.html', user=user)

    return render_template('evaluate.html', user=user)


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen(
        "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
            input=avi_file_path, output=output_name))
    return True


def genVideo(videopath):
    dest = videopath
    predicVal = []
    video = VideoFileClip(dest)
    start_time = 0
    end_time = 4
    video_duration = int(video.duration)
    vcap = cv2.VideoCapture(dest)
    if vcap.isOpened():
        # get vcap property
        w = int(vcap.get(3))
        h = int(vcap.get(4))

    j = 0

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 38, (w, h))

    for i in range(video_duration // 5):
        k = 0
        if os.path.exists("static/videos/" + str(j)):
            shutil.rmtree("static/videos/" + str(j))

        os.makedirs("static/videos/" + str(j))
        ffmpeg_extract_subclip(dest, start_time, end_time,
                               targetname="static/videos/" + str(j) + "/test.mp4")
        vid_capture = cv2.VideoCapture("static/videos/" + str(j) + "/test.mp4")
        predicVal.append(
            prediction('static/videos/' + str(j), 'static/violence/' + str(j) + 'preprocessedDataset/data/',
                       'static/violence/' + str(j) + 'preprocessedDataset'))


        frame_width = int(vid_capture.get(3))
        frame_height = int(vid_capture.get(4))

        while (True):
            ret, frame = vid_capture.read()
            if (ret == True):
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 80)
                fontScale = 1
                color = (0, 0, 255)
                thickness = 2
                if (predicVal[j] == 'True'):
                    if(k== 20):

                        suf = random.randint(0, 100)
                        filenamefull = "file" + str(suf)
                        today = date.today()
                        today1 = today.strftime("%B %d, %Y")
                        nows = datetime.now()
                        current_time = nows.strftime("%H:%M:%S")

                        ret, jpeg = cv2.imencode('.jpg', frame)
                        data = []
                        data.append(jpeg.tobytes())
                        frame1 = data[0]

                        fs.put(frame1, filename=filenamefull)
                        imgId = str(fs.put(frame1, filename=filenamefull))
                        insertDataLog(current_time, today1, imgId, "Violence","Recorded")

                    k+=1

                    color = (0, 0, 255)


                elif (predicVal[j] == 'False'):
                    color = (255, 0, 0)

                frame = cv2.putText(frame, predicVal[j], org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
            else:
                break

            out.write(frame)
        j += 1
        start_time += 4
        end_time += 4


def streamVid(videopath):
    vid_capture = cv2.VideoCapture(videopath)
    while (True):
        ret, frame = vid_capture.read()
        if (ret == True):
            ret, jpeg = cv2.imencode('.jpg', frame)
            data = []
            data.append(jpeg.tobytes())
            frame = data[0]
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        else:
            break




@app.route('/result', methods=['POST'])
def result():
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    target = os.path.join(APP_ROOT, "static/detectVideo/")
    target1 = os.path.join(APP_ROOT, "static/detectImage/")

    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    if os.path.exists(target1):
        shutil.rmtree(target1)
    os.makedirs(target1)

    if not os.path.isdir(target1):
        os.mkdir(target1)

    filename = ""
    for file in request.files.getlist("file"):
        filename = file.filename
        dest = "/".join([target, filename])
        detectedDest = "/".join([target, filename + "Detected"])
        print(dest)
        file.save(dest)
    genVideo(dest)

    shutil.rmtree('static/violence')

    return Response(streamVid('outpy.avi'), mimetype='multipart/x-mixed-replace; boundary=frame')


# YOLO Weapon Detection
@app.route('/show_weapon_detect', methods=['POST'])
def show_weapon_detect():
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    target = os.path.join(APP_ROOT, "static/detectVideo/")
    target1 = os.path.join(APP_ROOT, "static/detectImage/")

    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    if os.path.exists(target1):
        shutil.rmtree(target1)
    os.makedirs(target1)

    if not os.path.isdir(target1):
        os.mkdir(target1)

    filename = ""
    for file in request.files.getlist("file"):
        filename = file.filename
        dest = "/".join([target, filename])
        detectedDest = "/".join([target, filename + "Detected"])
        print(dest)
        file.save(dest)

    model = cv2.dnn.readNet('yolov3_training_final.weights',
                            'yolov3_training.cfg')

    classes = []
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()

    cap = cv2.VideoCapture(dest)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    # result = cv2.VideoWriter('static/detect.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

    iteration = 0
    while True:
        ret, image = cap.read()
        if (iteration == 2):
            r1 = random.random()
            r2 = random.random()
            r3 = random.uniform(7, 19)
            r4 = random.randint(1, 10)

            iteration = 0
            if ret == False:
                break
            img = image

            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            output_layer_names = model.getUnconnectedOutLayersNames()
            ########Forward Pass################
            layer_output = model.forward(output_layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_output:

                for prediction in output:
                    scores = prediction[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if (confidence > 0.1):
                        center_x = int(prediction[0] * width)
                        center_y = int(prediction[1] * height)
                        w = int(prediction[2] * width)
                        h = int(prediction[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
            if (len(indexes) != 0):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (label != "Human_face"):

                        confidence = str(round(confidences[i], 2))
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        suf = random.randint(0, 100)
                        filenamefull = "file" + str(suf)

                        ret, jpeg = cv2.imencode('.jpg', img)
                        data = []
                        data.append(jpeg.tobytes())
                        frame = data[0]
                        today = date.today()
                        today1 = today.strftime("%B %d, %Y")
                        nows = datetime.now()
                        current_time = nows.strftime("%H:%M:%S")

                        fs.put(frame, filename=filenamefull)
                        imgId = str(fs.put(frame, filename=filenamefull))
                        insertDataLog(current_time, today1, imgId,"Weapon","Recorded")

                        cv2.imwrite(
                            os.path.join(target1, str(r1) + str(r2) + str(r3) + str(r4) + "detectImage" + str(i) + ".jpg"),
                            img)

            image = img
            # result.write(image)

        iteration += 1

    cap.release()
    # result.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    images = os.listdir(os.path.join(app.static_folder, "detectImage"))

    return render_template("result.html", user=user, images=images, filename=filename)


@app.route('/dashboard')
def dashboard():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})

    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    cursor1 = camInfo.find({})
    for x in cursor1:
        if (x['name'] == "webcam"):
            cam = x
    det = Detection()
    # det.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")
    if user['role'] == "manager":
        return render_template('dashboard.html', user=user, cam=cam, det=det)
    else:
        return render_template('staff.html', user=user, cam=cam, det=det)


@app.route('/addStaff')
def addStaff():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    return render_template('addStaff.html', user=user)


@app.route('/removeMember')
def removeMember():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    return render_template('removeMember.html', user=user)


@app.route('/staffLogs')
def staffLogs():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    abc = staffLogInfo.find();
    return render_template('stafflogs.html', user=user, logs=abc)


@app.route('/updateDetails')
def updateDetails():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
                return render_template('updateDetails.html', user=user)

    return redirect(login)


@app.route('/manageRoles')
def manageRoles():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
                return render_template('manageRoles.html', user=user)

    return redirect(login)


@app.route('/userConfiguration')
def userConfiguration():
    print("yes")
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    return render_template('addRemoveUser.html', user=user)


# CRUD ROUTES

@app.route('/added', methods=["POST"])
def added():
    cursor = userInfo.find({})
    # if 'user_id' in session:
    for x in cursor:
        if ((x['username'] == request.form.get('username'))):
            return redirect(addStaff)
        if ((x['email'] == request.form.get('email'))):
            return redirect(addStaff)

    mem = Member()
    rep = mem.addMember(request)
    return rep
    # res=Member().addMember()
    # return render_template('added.html',res=res)


@app.route('/remove', methods=["POST"])
def remove():
    cursor = userInfo.find({})
    # if 'user_id' in session:
    for x in cursor:
        if ((x['username'] == request.form.get('username'))):
            mem = Member()
            rep = mem.removeMember(request)
            return rep

    # res=Member().addMember()
    # return render_template('added.html',res=res)
    return redirect(removeMember)


@app.route('/enableCam', methods=["POST"])
def enableCam():
    cursor1 = camInfo.find({})
    for x in cursor1:
        if (x['name'] == "webcam"):
            cam = x
    quer = {
        "name": "webcam"
    }

    c1 = Camera()
    enable = "true"
    rep = c1.updateCamera(quer, enable)
    d1 = Detection()
    rep = d1.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")

    return rep


@app.route('/disableCam', methods=["POST"])
def disableCam():
    cursor1 = camInfo.find({})
    for x in cursor1:
        if (x['name'] == "webcam"):
            cam = x
    quer = {
        "name": "webcam"
    }

    c1 = Camera()
    enable = "false"
    rep = c1.updateCamera(quer, enable)

    return rep


@app.route('/updateit', methods=["POST"])
def updateit():
    cursor = userInfo.find({})

    if (request.form.get('username') != g.user['username']):
        for x in cursor:
            if (x['username'] == request.form.get('username')):
                return redirect(updateDetails)

    elif (request.form.get('email') != g.user['email']):
        for x in cursor:
            if (x['email'] == request.form.get('email')):
                return redirect(updateDetails)

    queryUsername = {
        'username': request.form.get('username')
    }
    mem = Member()
    rep = mem.updateInfo(queryUsername, request)
    # res=Member().addMember()
    # return render_template('added.html',res=res)
    return rep


@app.route('/roleUpdate', methods=["POST"])
def roleUpdate():
    cursor = userInfo.find({})
    for x in cursor:
        if ((x['username'] == request.form.get('username'))):
            mem = Member()
            rep = mem.updateRole(request.form.get("username"), request.form.get("role"))

            return rep

    return redirect(manageRoles)


@app.route('/checkDetection', methods=["GET"])
def checkDetection():
    d1 = Detection()
    rep = d1.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")

    return rep


@app.route('/notifyManager', methods=["GET"])
def notifyManager():
    cursor1 = userInfo.find({})
    numbers = []
    for x in cursor1:
        numbers.append(x['phone'])

    d1 = Detection()
    # numbers = ['+923357957744']
    send_bulk_sms(numbers, "Detection Alert!!!")
    rep = d1.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")
    return rep


@app.route('/ignoreIt', methods=["GET"])
def ignoreIt():
    d1 = Detection()
    print("Detect")
    d1.setDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb", "false")
    rep = d1.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")
    return rep


@app.route('/informPolice', methods=["GET"])
def informPolice():
    d1 = Detection()
    numbers = ['+923046850743', '+923357957744', '+923174203526', '+923367957746']
    send_bulk_sms(numbers, "Red Alert at Bank#420 \n \n Thanks, \n OJO Smart Surveillance System")
    rep = d1.getDetectionState("9de1c199f1764b2b9afa12d2b5eeb8bb")
    return rep


@app.route('/newdash', methods=["GET"])
def newdash():
    return render_template("newdash.html")
