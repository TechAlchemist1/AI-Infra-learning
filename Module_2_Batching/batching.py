from collections import deque
from flask import Flask, jsonify, request,render_template   
from ollama import ChatResponse, chat
import time
from threading import Event
from Job import Job
import threading

server = Flask(__name__)

batchedRequest = deque()
waitingQueue = deque()
MAXBATCHSIZE = 10

''' 
We want to take the batch of requests and essentially ensure we wait for all requests to come in, 
and because llm responces are different in terms of computation time we need to check if we are done 
at each decode phase Intital appraoch will be continous batching
'''
def batch():
    # we dont start the process unless all the requests come in at the same time
    if len(waitingQueue) >= MAXBATCHSIZE:
        for _ in range(MAXBATCHSIZE):
            batchedRequest.append(waitingQueue.popleft())
        while batchedRequest:
            userRequestObj = batchedRequest.popleft()
            response: ChatResponse = chat(
                model="qwen2.5:3b",
                messages=[
                    {
                        "role": "user",
                        "content": userRequestObj.prompt
                    }
                ],
                options={"num_thread": 8},
                keep_alive="30m",
            )
            userRequestObj.result = response
            userRequestObj.done.set() 

# create a watcher of the batch
def batchWatcher():
    # determine if batch is the size needed
    while 1:

        if len(waitingQueue) >= MAXBATCHSIZE:
            batch()
        time.sleep(0.01)

@server.route("/spice", methods=["POST"])
def postGreet():

    userRequest = request.get_json()
    # make a job object for each call and wait 
    jobEvent = Job(userRequest["prompt"])
    # add this event to the waiting queue
    waitingQueue.append(jobEvent)
    # make the watcher thread 
    
    jobEvent.done.wait()
    # once its done return the request from its object call the method
    # make the responce return 
    return jsonify({
        "response": jobEvent.result["message"]["content"],
        "prompt_tokens": jobEvent.result["prompt_eval_count"],
        "output_tokens": jobEvent.result["eval_count"]
    })
    

 
@server.route("/")
def home():
    return render_template("index.html")

@server.route("/greet")
def hello():
    # in our greet we make a call to our ollama model
    response: ChatResponse = chat(model="qwen2.5:3b", messages=[
        {
            'role' : 'user',
            'content': 'greet me in somali',
        },
    ])

    return response['message']['content']
if __name__ == "__main__":
    watcherThread = threading.Thread(target=batchWatcher,daemon=True)
    watcherThread.start()
    server.run(debug=True)