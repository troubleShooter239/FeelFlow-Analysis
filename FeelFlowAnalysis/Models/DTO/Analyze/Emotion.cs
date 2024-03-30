using System.Collections;
using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class Emotion : IEnumerable<KeyValuePair<string, float>>
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

    public IEnumerator<KeyValuePair<string, float>> GetEnumerator()
    {
        yield return new KeyValuePair<string, float>("angry", Angry);
        yield return new KeyValuePair<string, float>("disgust", Disgust);
        yield return new KeyValuePair<string, float>("fear", Fear);
        yield return new KeyValuePair<string, float>("happy", Happy);
        yield return new KeyValuePair<string, float>("sad", Sad);
        yield return new KeyValuePair<string, float>("surprise", Surprise);
        yield return new KeyValuePair<string, float>("neutral", Neutral);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}