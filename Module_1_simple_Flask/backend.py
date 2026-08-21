# The goal of this is simple we make a call to a local ollama model
# get the returned output and print it

from flask import Flask, jsonify, request,render_template   
from ollama import ChatResponse, chat

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

@server.route("/spice", methods=['POST'])
def postGreet():
    # parse the data? and make the chat object 
    data = request.get_json()
    # now we have a prompt field {prompt : "do xyz"}
    response:ChatResponse = chat(model="qwen2.5:3b",messages=[
        {
            'role':'user',
            'content':data["prompt"],

        },
    ])
    return jsonify({"response": response['message']['content']})

if __name__ == "__main__":
    server.run(debug=True)