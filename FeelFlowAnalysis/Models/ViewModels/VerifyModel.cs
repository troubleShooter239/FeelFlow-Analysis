using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.ViewModels;

public class VerifyModel : IVerifyModel
{
    [JsonPropertyName("b64_img1")]
    public string B64img1 { get; set; } = string.Empty;
    [JsonPropertyName("b64_img2")]
    public string B64img2 { get; set; } = string.Empty;
    [JsonPropertyName("r_model_name")]
    public string RModelName { get; set; } = string.Empty;
    [JsonPropertyName("distance_metric")]
    public string DistanceMetric { get; set; } = string.Empty;
}
