using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class Race
{
    [JsonPropertyName("asian")]
    public float Asian { get; set; }

    [JsonPropertyName("indian")]
    public float Indian { get; set; }

    [JsonPropertyName("black")]
    public float Black { get; set; }

    [JsonPropertyName("white")]
    public float White { get; set; }

    [JsonPropertyName("middle_eastern")]
    public float MiddleEastern { get; set; }

    [JsonPropertyName("latino_hispanic")]
    public float LatinoHispanic { get; set; }
}
