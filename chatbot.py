import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import http.client

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'C:/Users/hp/Desktop/chatbot_project_ai/intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure what you mean. Can you try asking in a different way?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if tag == 'weather':
                return fetch_weather()
            else:
                return random.choice(i['responses'])
    return "I'm not sure what you mean. Can you try asking in a different way?"

def fetch_weather():
    conn = http.client.HTTPSConnection("open-weather13.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "2ddb618d88msh1f24f4c76f46d54p14984cjsne04774a0a5c8",
        'x-rapidapi-host': "open-weather13.p.rapidapi.com"
    }
    # Corrected endpoint for Marrakech, Morocco
    conn.request("GET", "/city/Marrakech/EN", headers=headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    weather_data = json.loads(data)
    
    if 'main' in weather_data and 'weather' in weather_data:
        temperature_kelvin = weather_data['main']['temp']
        temperature_celsius = round(temperature_kelvin - 273.15, 2)
        description = weather_data['weather'][0]['description']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        
        response = f"The weather in {weather_data['name']} is currently {description}. "
        response += f"The temperature is {temperature_celsius}°C, "
        response += f"humidity is {humidity}%, and wind speed is {wind_speed} m/s."
        return response
    else:
        return "I'm sorry, I couldn't retrieve the weather information for Marrakech at the moment. Please try again later."

print("GO! Bot is running!")

while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
