using System.Net.Http.Json;
using System.Text.Json;
using AspireApp1.Web.Models;
namespace AspireApp1.Web;

public class StockPredictionApiClient
{
  private readonly HttpClient _httpClient;

  public StockPredictionApiClient(HttpClient httpClient)
  {
    _httpClient = httpClient;
  }

  public async Task<StockPredictionResponse> PredictStockPriceAsync(StockPredictionRequest request, CancellationToken cancellationToken = default)
  {
    string json = JsonSerializer.Serialize(request);

// Print to console (in Rider, it appears in the debug console)
    Console.WriteLine("Request JSON:");
    Console.WriteLine(json);
    
    var response = await _httpClient.PostAsJsonAsync("/stockprediction", request, cancellationToken);

    if (response.IsSuccessStatusCode)
    {
      var result = await response.Content.ReadFromJsonAsync<StockPredictionResponse>(cancellationToken: cancellationToken);
      return result ?? new StockPredictionResponse { Message = "No response from server." };
    }
    else
    {
      throw new HttpRequestException($"Request failed with status code {response.StatusCode}");
    }
  }
}

