from threading import Event

class Job:
    def __init__(self, prompt):
        self.prompt = prompt
        self.result = None
        self.done = Event()