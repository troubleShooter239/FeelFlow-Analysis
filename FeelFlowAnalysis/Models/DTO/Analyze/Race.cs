using System.Collections;
using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Analyze;

public class Race : IEnumerable<KeyValuePair<string, float>>
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

    public IEnumerator<KeyValuePair<string, float>> GetEnumerator()
    {
        yield return new KeyValuePair<string, float>("asian", Asian);
        yield return new KeyValuePair<string, float>("indian", Indian);
        yield return new KeyValuePair<string, float>("black", Black);
        yield return new KeyValuePair<string, float>("white", White);
        yield return new KeyValuePair<string, float>("middle_eastern", MiddleEastern);
        yield return new KeyValuePair<string, float>("latino_hispanic", LatinoHispanic);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
