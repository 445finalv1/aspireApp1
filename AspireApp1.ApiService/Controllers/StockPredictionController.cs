using System.Security.Cryptography.X509Certificates;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AspireApp1.ApiService.Models;

namespace AspireApp1.ApiService.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class StockPredictionController : ControllerBase
    {
        [HttpPost]
        public IActionResult Predict([FromBody] StockPredictionRequest request)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest("Invalid data.");
            }

            try
            {
                // Load the ONNX model for the specified company
                var modelPath = Path.Combine("Models", $"{request.CompanyName}_model.onnx");
                if (!System.IO.File.Exists(modelPath))
                {
                    return NotFound($"Model for {request.CompanyName} not found.");
                }

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

                return Ok(response);
            }
            catch (Exception ex)
            {
                // Handle exceptions
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
        }

        private List<NamedOnnxValue> PrepareInputData(StockPredictionRequest request, string inputName)
        {
            var inputFeatures = new List<float>();

            // Add lag features for recent stock prices
            // foreach (var price in request.RecentStockPrices)
            // {
            //     inputFeatures.Add((float)price);
            // }

            for (int i = 0; i < 5; i++)
            {
                inputFeatures.Add((float)request.RecentStockPrices[i]);
            }
            
            // Add moving averages (you may need to compute them here)
            var ma5 = request.RecentStockPrices.TakeLast(5).Average();
            var ma10 = request.RecentStockPrices.Average();
            inputFeatures.Add((float)ma5);
            inputFeatures.Add((float)ma10);

            // Add related company prices
            // foreach (var price in request.RelatedCompanyPrices.Values)
            // {
            //     inputFeatures.Add((float)price);
            // }

            for (int i = 0; i < 3; i++)
            {
                inputFeatures.Add((float)request.RelatedCompanyPrices.Values.ElementAt(i));
            }
            
            // Add economic indicators
            // foreach (var indicatorValues in request.EconomicIndicators.Values)
            // {
            //     foreach (var value in indicatorValues)
            //     {
            //         inputFeatures.Add((float)value);
            //     }
            // }
            
            Console.WriteLine("Size of input features is: " + inputFeatures.ToArray().Length);

            // Convert input features to tensor
            var tensor = new DenseTensor<float>(inputFeatures.ToArray(), new[] { 1, inputFeatures.Count });

            // Create named ONNX value
            var inputData = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            };
            Console.WriteLine("Size of input data is: " + inputData.ToArray().Length);
            return inputData;
        }
    }
}