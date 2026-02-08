import joblib

model = joblib.load("model/classifier.pkl")

def predict_ticket(text):
    return model.predict([text])[0]

print("Smart Ticket Classifier")

while True:
    ticket = input("Enter issue (or 'exit'): ")
    
    if ticket.lower() == "exit":
        break
    
    result = predict_ticket(ticket)
    print("Predicted Category:", result)
