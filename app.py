import cv2
import time
import os
import uuid

import pymongo
import numpy as np
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

import threading
from camera import VideoCam



client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
mydb = client['ojo']

userInfo=mydb.userInfo

username1 = ""
password1 = ""
alert=False
class Member:

    def start_session(self,member):
        session['logged_in']=True


        return jsonify(member),200

    def signIn(self,username,password):
        member = {
            "username":username,
            "password": password,
        }

        if(userInfo.find_one(member)):
            return True
        else: return False



    def addMember(self,request):
        member = {
            "_id":uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "username": request.form.get('username'),
            "password": request.form.get('password'),
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

    def updateInfo(self, query,request):
        member = {

            "$set":{
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "username": request.form.get('username'),
            "password": request.form.get('password'),
            "role": request.form.get('role')
            }


        }

        userInfo.update_one(query, member)


        return jsonify(member), 200

app = Flask(__name__)
app.secret_key = 'somesecretkeythatonlyishouldknow'
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.before_request
def before_request():
    g.user = None
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if(x['_id']==session['user_id']):
                user=x
                g.user=user



@app.route('/login', methods=['GET', 'POST'])
def login():
    session.pop('user_id', None)
    if request.method == 'POST':
        session.pop('user_id', None)

        username1 = request.form['username']
        password1 = request.form['password']

        userData=userInfo.find_one({'username':username1})
        if(userData):
            if(userData['password']==password1):
                session['user_id'] = userData['_id']
                return redirect(url_for('dashboard'))
            return render_template('login.html')

        return render_template('login.html')


    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


def gen():

    cap = cv2.VideoCapture(0)

    model = cv2.dnn.readNet('yolov3_training_final.weights',
                            'yolov3_training.cfg')

    classes = []
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()

    while True:
        ret, image = cap.read()
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        colors = np.random.uniform(0, 255, size=(len(indexes), 3))
        if (len(indexes) != 0):
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                confidence = str(round(confidences[i], 2))
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img[:, :, 0] = 0
                img[:, :, 1] = 0
                cv2.imshow('R-RGB', img)

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        image=img

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        frame = data[0]
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



    cap.release()
    cv2.destroyWindow()


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def main():
    return redirect(url_for('dashboard'))

@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')


@app.route('/result',methods=['POST'])
def result():
    target=os.path.join(APP_ROOT,"videos/")
    print(target)

    if not os.path.isdir(target):
       os.mkdir(target)

    for file in request.files.getlist("file"):

        filename=file.filename
        dest="/".join([target,filename])
        detectedDest = "/".join([target, filename+"Detected"])
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

    result = cv2.VideoWriter('static/detect.mp4',
                             cv2.VideoWriter_fourcc(*'MP4V'),
                             25, size)
    while True:
        ret, image = cap.read()
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
                if (confidence > 0.2):
                    center_x = int(prediction[0] * width)
                    center_y = int(prediction[1] * height)
                    w = int(prediction[2] * width)
                    h = int(prediction[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        if (len(indexes) != 0):
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        image = img
        result.write(image)
    cap.release()
    result.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return render_template("result.html")


@app.route('/dashboard')
def dashboard():
    if not g.user:
        return redirect(url_for('login'))

    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    if user['role'] == "manager":
        return render_template('dashboard.html', alert=alert)
    else:
        return render_template('staff.html', alert=alert)


@app.route('/addStaff')
def addStaff():
    if not g.user:
        return redirect(url_for('login'))
    return render_template('addStaff.html')



@app.route('/removeMember')
def removeMember():
    if not g.user:
        return redirect(url_for('login'))
    return render_template('removeMember.html')


@app.route('/updateDetails')
def updateDetails():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    return render_template('updateDetails.html',user=user)

@app.route('/userConfiguration')
def userConfiguration():

    return render_template('addRemoveUser.html')


#CRUD ROUTES

@app.route('/added',methods=["POST"])
def added():
    mem=Member()
    rep=mem.addMember(request)
    #res=Member().addMember()
    #return render_template('added.html',res=res)
    return rep


@app.route('/remove',methods=["POST"])
def remove():
    mem=Member()
    rep=mem.removeMember(request)
    #res=Member().addMember()
    #return render_template('added.html',res=res)
    return rep



@app.route('/updateit',methods=["POST"])
def updateit():
    mem=Member()
    cursor = userInfo.find({})

    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    queryUsername={
        'username':user['username']
    }
    rep=mem.updateInfo(queryUsername,request)
    #res=Member().addMember()
    #return render_template('added.html',res=res)
    return rep