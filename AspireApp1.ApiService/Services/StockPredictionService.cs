namespace AspireApp1.ApiService.Services;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;

public class StockPredictionService
{
  private readonly ConcurrentDictionary<string, InferenceSession> _modelSessions;

  public StockPredictionService()
  {
    _modelSessions = new ConcurrentDictionary<string, InferenceSession>();
  }

  private InferenceSession GetModelSession(string modelPath)
  {
    return _modelSessions.GetOrAdd(modelPath, path => new InferenceSession(path));
  }

  public float Predict(string stockSymbol, float[] features)
  {
    // Build the model path
    string modelPath = Path.Combine("Models", $"{stockSymbol}_model.onnx");

    // Check if the model file exists
    if (!File.Exists(modelPath))
      throw new FileNotFoundException($"Model file not found for stock symbol: {stockSymbol}");

    var session = GetModelSession(modelPath);

    // Prepare input data
    var inputs = new List<NamedOnnxValue>
    {
      NamedOnnxValue.CreateFromTensor("float_input", new DenseTensor<float>(features, new int[] { 1, features.Length }))
    };

    // Run inference
    using var results = session.Run(inputs);

    // Extract and return the prediction
    var output = results.First().AsEnumerable<float>().First();
    return output;
  }
}