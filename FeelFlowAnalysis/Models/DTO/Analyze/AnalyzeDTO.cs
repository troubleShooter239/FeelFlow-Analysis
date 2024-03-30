using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class AnalyzeDTO
{
    [JsonPropertyName("region")]
    public Region? Region { get; set; }
    [JsonPropertyName("face_confidence")]
    public float FaceConfidence { get; set; }
    [JsonPropertyName("age")]
    public int Age { get; set; }
    [JsonPropertyName("emotion")]
    public Emotion? Emotion { get; set; }
    [JsonPropertyName("dominant_emotion")]
    public string? DominantEmotion { get; set; }
    [JsonPropertyName("gender")]
    public Gender? Gender { get; set; }
    [JsonPropertyName("dominant_gender")]
    public string? DominantGender { get; set; }
    [JsonPropertyName("race")]
    public Race? Race { get; set; }
    [JsonPropertyName("dominant_race")]
    public string? DominantRace { get; set; }
}
