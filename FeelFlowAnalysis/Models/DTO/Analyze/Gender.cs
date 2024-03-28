using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class Gender
{
    [JsonPropertyName("woman")]
    public float Woman { get; set; }
    [JsonPropertyName("man")]
    public float Man { get; set; }
}
