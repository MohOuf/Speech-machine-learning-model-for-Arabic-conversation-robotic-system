 # Speech-machine-learning-model-for-conversation-robotic-system
## Two Simple methods to create Chat-bot 
### 1. Create Chat-bot using (IBM Watson Chat Bot)
* 1.01: open the following Link : https://cloud.ibm.com/catalog/services/watson-assistant
* 1.02: Create an account 
* 1.03: Click in <Create an assistant>
* 1.04: Start create the *skills* which is beneficial in modifying the Chat bot
* 1.05: Create a *dialog* skill in order to create *intents* which is important in organizing the dialog and the response for the Chat-bot
Link for my Chat-Bot ---> https://web-chat.global.assistant.watson.cloud.ibm.com/preview.html?region=eu-gb&integrationID=49917c4e-5a6f-4941-b434-89ecb7b5be2f&serviceInstanceID=cadcead9-1bba-4bb1-880b-61fda8ddef9d 

### 2. Create Chat-bot using python (Miniconda) 
* 2.01: Download Miniconda from the following link: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblhtT0NsQVA3TGFKOThlY2tFbk0yU3BXcDA1QXxBQ3Jtc0ttXzN1RUJLZFBzbHpmZnlnbk16SDh4cEEyYXoxamtvbDhndVBtRU1vWnJSdDBHc2tpWWh5ZWQtRHdjWE9ZbVdkV1Bfb004Q1h3SlVYYTNlN0t0MzkwREhhVjJTdmRacV9RWFAxRDZ4ZkUzbkRHLWVVTQ&q=https%3A%2F%2Fdocs.conda.io%2Fen%2Flatest%2Fminiconda.html
* 2.02: Open the *Miniconda terminal* and write in the first line: conda create --name (the name of your chat-bot) python (the version number) *to create the virtual environment*
* 2.03: After creating the virtual environment write: conda install -c anaconda pip *to download the packages*
* 2.04: Activate the Chat-bot by typing: activate (the name of your chat-bot) 
* 2.05: Download the following packages in the anaconda (TenserFlow, tflearn, nltk) to use their libraries
  
###Code: 
 ```
 import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow.compat.v1 as tf
import tflearn 
import random
import json
import pickle

from time import sleep

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = [] 
    for intent in data ["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)


        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.reset_default_graph

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat():
    print("السلام عليكم انا روبوت الشات كيف اقدر اخدمك")
    while True:
        inp = input("انت: ")
        if inp.lower() == "خروج":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            print(Bot)
        else:
            print("الصراحه ما فهمت عليك ممكن تعيد")
chat()
 ```
 ### intents Code : 
 ```
 {"intents": [
        {"tag": "greeting",
         "patterns": ["هلا هلا", "هلا والله كيف الحال", "هلا " , " السلام عليكم"],
         "responses": ["اهلاااا", "مرحبا", "هلا مين,", "كيف اقدر اخدمك"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["مع السلامه", "الوداع", "سلام"],
         "responses": ["نشوفك بعدين", "مع السلامه", "وقت ثاني ان شاء الله"],
         "context_set": ""
        },
        {"tag": "age",
         "patterns": ["كم عمرك"],
         "responses": ["انا روبوت العمر عندي شيء افتراضي ", "كيف؟"],
         "context_set": ""
        },
        {"tag": "name",
         "patterns": ["كيف اقدر اناديك", "اسمك", "ايش اسمك", "مين انت"],
         "responses": ["عوف الروبوت في الخدمه", "انا عوف"],
         "context_set": ""
        },
        {"tag": "help",
         "patterns": ["ممكن سؤال", "ايش فايدتك", "سؤال ", "عندي سؤال"],
         "responses": ["انا موجود هنا لخدمتك"],
         "context_set": ""
        }
   ]
}
 ```
to Lunch the bot type : python main.py 
 
