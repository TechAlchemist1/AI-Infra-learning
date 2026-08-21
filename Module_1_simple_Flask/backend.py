# The goal of this is simple we make a call to a local ollama model
# get the returned output and print it

from flask import Flask, jsonify, request,render_template   
from ollama import ChatResponse, chat
import time


# we need to make an instance of flask server 

server = Flask(__name__)

# Server starts and listens at port xxx -> we go there and to does something 
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

@server.route("/spice", methods=["POST"])
def postGreet():
    start = time.time()

    data = request.get_json()

    response: ChatResponse = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": data["prompt"]
            }
        ],
        # num_thread=8: benchmarks/baseline.py showed Ollama's default thread
        # heuristic on this 8-core CPU only uses ~4 threads, leaving ~28%
        # decode throughput on the table (22.7 -> 29 tok/s). keep_alive=30m
        # keeps the model resident between requests so a single interactive
        # user doesn't eat a multi-second reload after a short idle gap.
        options={"num_thread": 8},
        keep_alive="30m",
    )

    latency = time.time() - start
    NS = 1_000_000_000


    return jsonify({
        "response": response["message"]["content"],
        "latency": latency,
        "total_duration": response["total_duration"],
        "load_duration": response["load_duration"] / NS,

        "prompt_tokens": response["prompt_eval_count"],
        "prompt_duration": response["prompt_eval_duration"] / NS,

        "output_tokens": response["eval_count"],
        "decode_duration": response["eval_duration"] /NS
    })
if __name__ == "__main__":
    server.run(debug=True)