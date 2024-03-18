using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.ViewModels;

public class MetadataViewModel
{
    [JsonPropertyName("b64_img")]
    public string B64img { get; set; } = string.Empty;
}
