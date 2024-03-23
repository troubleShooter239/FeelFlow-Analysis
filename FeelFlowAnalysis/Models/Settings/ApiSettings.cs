namespace FeelFlowAnalysis.Models.Settings;

public class ApiSettings : IApiSettings
{
    public string ApiUrl { get; set; } = string.Empty;
    public string Analyze { get; set; } = string.Empty;
    public string Metadata { get; set; } = string.Empty;
    public string Verify { get; set; } = string.Empty;
}
