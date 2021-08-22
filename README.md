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
 
to Lunch the bot type : python main.py 
 
