namespace AspireApp1.Web
{
  public class CompanyListApiClient
  {
    private readonly HttpClient _httpClient;

    /// <summary>
    /// Initializes a new instance of the <see cref="CompanyListApiClient"/> class.
    /// </summary>
    /// <param name="httpClient">The <see cref="HttpClient"/> instance used to send HTTP requests.</param>
    public CompanyListApiClient(HttpClient httpClient)
    {
      _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
    }

    /// <summary>
    /// Retrieves the list of company names from the /companylist endpoint.
    /// </summary>
    /// <param name="maxItems">The maximum number of items to retrieve. Defaults to 100.</param>
    /// <param name="cancellationToken">A token to monitor for cancellation requests.</param>
    /// <returns>An array of company name strings.</returns>
    public async Task<string[]> GetCompanyListAsync(int maxItems = 100, CancellationToken cancellationToken = default)
    {
      // Send a GET request to the /companylist endpoint and deserialize the response as a string array
      var companies = await _httpClient.GetFromJsonAsync<string[]>("/companylist", cancellationToken);

      if (companies == null)
      {
        return Array.Empty<string>();
      }

      // If maxItems is specified and less than the total count, return the subset
      if (companies.Length > maxItems)
      {
        return companies.AsSpan(0, maxItems).ToArray();
      }

      return companies;
    }
  }
}