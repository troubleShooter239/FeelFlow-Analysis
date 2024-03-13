using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models;

public class AnalyzeModel : IAnalyzeModel
{
    [JsonPropertyName("b64_img")]
    public string B64img { get; set; } = string.Empty;
    [JsonPropertyName("actions")]
    public Dictionary<string, bool> Actions { get; set; } = [];
}
