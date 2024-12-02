namespace AspireApp1.ApiService.Models;

public class StockPredictionRequest
{
  public string CompanyName { get; set; }
  public List<decimal> RecentStockPrices { get; set; } // Last 10 days
  public Dictionary<string, decimal> RelatedCompanyPrices { get; set; } // 5 companies
  public Dictionary<string, List<decimal>> EconomicIndicators { get; set; } // 4 indicators, last 2 days
}