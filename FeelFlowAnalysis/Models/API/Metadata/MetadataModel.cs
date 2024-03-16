using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.API.Metadata;

public class MetadataModel : IMetadataModel
{
    [JsonPropertyName("b64_img")]
    public string B64img { get; set; } = string.Empty;
}
