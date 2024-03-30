using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class Emotion
{
    [JsonPropertyName("angry")]
    public float Angry { get; set; }

    [JsonPropertyName("disgust")]
    public float Disgust { get; set; }

    [JsonPropertyName("fear")]
    public float Fear { get; set; }

    [JsonPropertyName("happy")]
    public float Happy { get; set; }

    [JsonPropertyName("sad")]
    public float Sad { get; set; }

    [JsonPropertyName("surprise")]
    public float Surprise { get; set; }

    [JsonPropertyName("neutral")]
    public float Neutral { get; set; }
}