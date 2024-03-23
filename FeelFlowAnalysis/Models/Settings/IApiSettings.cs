namespace FeelFlowAnalysis.Models.Settings;

public interface IApiSettings
{
    string ApiUrl { get; set; }
    string Analyze { get; set; }
    string Metadata { get; set; }
    string Verify { get; set; }
}
