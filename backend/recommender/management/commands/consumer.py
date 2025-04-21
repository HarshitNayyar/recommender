import stomp
import time
from django.core.management.base import BaseCommand
from recommender.models import Messages

class ArtemisListener(stomp.ConnectionListener):
    def on_message(self, frame):
        print("Ricevuto messaggio:", frame.body)
        Messages.objects.create(message=frame.body)

class Command(BaseCommand):
    help = "Consume messages from Artemis via STOMP"

    def handle(self, *args, **kwargs):
        conn = stomp.Connection([('localhost', 61645)])  # ✅ porta corretta
        conn.set_listener('', ArtemisListener())
        conn.connect('admin', 'admin', wait=True)
        conn.subscribe(destination='catalog', id=1, ack='auto')  # ✅ senza '/queue/'

        self.stdout.write(self.style.SUCCESS("In ascolto sulla coda 'catalog'..."))

        while True:
            time.sleep(1)