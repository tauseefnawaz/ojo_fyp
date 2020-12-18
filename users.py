from flask import Flask, jsonify, request
import uuid
from app import userInfo


class Member:

    def addMember(self):
        member = {
            "name": uuid.uuid4().hex,
            "email": request.form.get('name'),
            "username": request.form.get('username'),
            "password": request.form.get('password'),
            "role": request.form.get('role')
        }
        userInfo.insert_one(member)

        return jsonify(member), 200
