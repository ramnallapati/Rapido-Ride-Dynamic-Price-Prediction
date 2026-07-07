from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load Preprocessor
with open("dynamic_price_preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# Load Model
with open("dynamic_price_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = pd.DataFrame({

        "Number_of_Riders":[int(request.form["Number_of_Riders"])],

        "Number_of_Drivers":[int(request.form["Number_of_Drivers"])],

        "Location_Category":[request.form["Location_Category"]],

        "Customer_Loyalty_Status":[request.form["Customer_Loyalty_Status"]],

        "Number_of_Past_Rides":[int(request.form["Number_of_Past_Rides"])],

        "Average_Ratings":[float(request.form["Average_Ratings"])],

        "Time_of_Booking":[request.form["Time_of_Booking"]],

        "Vehicle_Type":[request.form["Vehicle_Type"]],

        "Expected_Ride_Duration":[int(request.form["Expected_Ride_Duration"])]

    })

    transformed_data = preprocessor.transform(data)

    selected_features = ['num__Number_of_Riders', 'num__Number_of_Drivers','num__Average_Ratings', 'num__Expected_Ride_Duration', 'ohe__Time_of_Booking_Evening', 'ohe__Vehicle_Type_Premium']
    
    train_df = pd.DataFrame(transformed_data,columns=preprocessor.get_feature_names_out())
    x_train_best_features = train_df[selected_features]   
    prediction = model.predict(x_train_best_features)[0]

    return render_template(
        "index.html",
        prediction=f"Estimated Ride Price : ₹ {prediction:.2f}"
    )


if __name__ == "__main__":
    app.run()