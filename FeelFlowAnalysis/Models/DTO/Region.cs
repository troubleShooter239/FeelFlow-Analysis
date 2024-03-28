using System.Collections;
using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO;

public class Region : IEnumerable<KeyValuePair<string, int>>
{
    [JsonPropertyName("x")]
    public int X { get; set; }

    [JsonPropertyName("y")]
    public int Y { get; set; }

    [JsonPropertyName("w")]
    public int W { get; set; }

    [JsonPropertyName("h")]
    public int H { get; set; }

    public IEnumerator<KeyValuePair<string, int>> GetEnumerator()
    {
        yield return new KeyValuePair<string, int>("x", X);
        yield return new KeyValuePair<string, int>("y", Y);
        yield return new KeyValuePair<string, int>("w", W);
        yield return new KeyValuePair<string, int>("h", H);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}