import os
import json
from twilio.rest import Client

ACCOUNT_SID = 'ACb16ddebb35843cc193df64f2b9bc9356'
AUTH_TOKEN = '097ddd4c040cd474abbfee2aaceea8fb'
NOTIFY_SERVICE_SID = 'ISf7588634046fe7c65b86bc9e9b10c7c9'

client = Client(ACCOUNT_SID, AUTH_TOKEN)


def send_bulk_sms(numbers, body):
    bindings = list(map(lambda number: json.dumps({'binding_type': 'sms', 'address': number}), numbers))
    print("=====> Surveillance Alert :>", bindings, "<: =====")
    notification = client.notify.services(NOTIFY_SERVICE_SID).notifications.create(
        to_binding=bindings,
        body=body
    )

    print(notification.body)
