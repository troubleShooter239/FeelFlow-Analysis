using System.Collections;
using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class AnalyzeDTO : IEnumerable<KeyValuePair<string, object>>
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

    public IEnumerator<KeyValuePair<string, object>> GetEnumerator()
    {
        yield return new KeyValuePair<string, object>("region", Region);
        yield return new KeyValuePair<string, object>("face_confidence", FaceConfidence);
        yield return new KeyValuePair<string, object>("age", Age);
        yield return new KeyValuePair<string, object>("emotion", Emotion);
        yield return new KeyValuePair<string, object>("dominant_emotion", DominantEmotion);
        yield return new KeyValuePair<string, object>("gender", Gender);
        yield return new KeyValuePair<string, object>("dominant_gender", DominantGender);
        yield return new KeyValuePair<string, object>("race", Race);
        yield return new KeyValuePair<string, object>("dominant_race", DominantRace);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
