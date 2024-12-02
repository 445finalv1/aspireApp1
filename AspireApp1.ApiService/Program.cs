using AspireApp1.ApiService.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire components.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseExceptionHandler();

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

// New stockPrediction endpoint
app.MapPost("/stockprediction", (StockPredictionRequest request) =>
{
    if (request == null)
    {
        return Results.BadRequest("Invalid data.");
    }

    // Load the ONNX model for the specified company
    var modelPath = Path.Combine("Data", "Models", $"{request.CompanyName}_model.onnx");
    if (!System.IO.File.Exists(modelPath))
    {
        return Results.NotFound($"Model for {request.CompanyName} not found.");
    }

    try
    {
        using var session = new InferenceSession(modelPath);

        // Get the input name from the session
        var inputName = session.InputMetadata.Keys.First();

        // Prepare the input data
        var inputData = PrepareInputData(request, inputName);

        // Run inference
        using var results = session.Run(inputData);

        // Extract the prediction
        var predictedPrice = results.First().AsEnumerable<float>().First();

        // Return the response
        var response = new StockPredictionResponse
        {
            PredictedPrice = (decimal)predictedPrice,
            Message = "Prediction successful."
        };

        return Results.Ok(response);
    }
    catch (Exception ex)
    {
        // Handle exceptions
        return Results.Problem($"Internal server error: {ex.Message}");
    }

    // Local function to prepare input data
    List<NamedOnnxValue> PrepareInputData(StockPredictionRequest request, string inputName)
    {
        var inputFeatures = new List<float>();

        // Add lag features for recent stock prices
        foreach (var price in request.RecentStockPrices)
        {
            inputFeatures.Add((float)price);
        }

        // Add moving averages
        var ma5 = request.RecentStockPrices.TakeLast(5).Average();
        var ma10 = request.RecentStockPrices.Average();
        inputFeatures.Add((float)ma5);
        inputFeatures.Add((float)ma10);

        // Add related company prices
        foreach (var price in request.RelatedCompanyPrices.Values)
        {
            inputFeatures.Add((float)price);
        }

        // Add economic indicators
        foreach (var indicatorValues in request.EconomicIndicators.Values)
        {
            foreach (var value in indicatorValues)
            {
                inputFeatures.Add((float)value);
            }
        }

        // Convert input features to tensor
        var tensor = new DenseTensor<float>(inputFeatures.ToArray(), new[] { 1, inputFeatures.Count });

        // Create named ONNX value
        var inputData = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        return inputData;
    }
});

app.MapGet("/weatherforecast", () =>
{
    var forecast = Enumerable.Range(1, 5).Select(index =>
        new WeatherForecast
        (
            DateOnly.FromDateTime(DateTime.Now.AddDays(index)),
            Random.Shared.Next(-20, 55),
            summaries[Random.Shared.Next(summaries.Length)]
        ))
        .ToArray();
    return forecast;
});

app.MapDefaultEndpoints();

app.Run();

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}