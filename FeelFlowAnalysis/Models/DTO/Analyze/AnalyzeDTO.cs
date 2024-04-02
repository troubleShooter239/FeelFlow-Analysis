using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class AnalyzeDTO
{
    [JsonPropertyName("region")]
    public required Region Region { get; set; }
    [JsonPropertyName("face_confidence")]
    public float FaceConfidence { get; set; }
    [JsonPropertyName("age")]
    public int Age { get; set; }
    [JsonPropertyName("emotion")]
    public required Emotion Emotion { get; set; }
    [JsonPropertyName("dominant_emotion")]
    public required string DominantEmotion { get; set; }
    [JsonPropertyName("gender")]
    public required Gender Gender { get; set; }
    [JsonPropertyName("dominant_gender")]
    public required string DominantGender { get; set; }
    [JsonPropertyName("race")]
    public required Race Race { get; set; }
    [JsonPropertyName("dominant_race")]
    public required string DominantRace { get; set; }
}
