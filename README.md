### HSMR
HSMR is a stock prediction web application using the .NET Aspire framework and models trained off of a combination of random forest regressor, recursive feature elimination, and other machine-learning concepts learned in CMPSC 445 to predict stock values.

## Setup
There are two key steps of this application, training the models and running the web application. For starters, clone the repository using:
```console
git clone https://github.com/445finalv1/AspireApp1.git
```

Then, run the testMain.py in the trainModels folder inside of the AspireApp1.ApiService folder, or run the following command:

```console
python AspireApp1.ApiService/trainingModels/testMain.py
```

This will run for about twenty minutes, or shorter or longer depending on your computer. This will train all the models. The downloaded repository however will also have trained models in there.
Once the models are trained, move them into "AspireApp1.ApiService/Data/Models".

From there, you will need to install the .NET framework and .NET Aspire. links for installing for [Visual Studio and VS Code](https://learn.microsoft.com/en-us/dotnet/aspire/fundamentals/setup-tooling?tabs=linux&pivots=visual-studio) and [JetBrains Rider](https://blog.jetbrains.com/dotnet/2024/02/19/jetbrains-rider-and-the-net-aspire-plugin/) are listed here.

## Running the Application

From here, you should be able to select "Build and Run" in your IDE of choice, and the application should run! Navigate to the stock predictor page and you can observe the API call for the list of companies being executed. Select one from the drop-down menu, fill in data and submit a request.
