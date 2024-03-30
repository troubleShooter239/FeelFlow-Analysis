using System.Text.Json.Serialization;

namespace FeelFlowAnalysis.Models.DTO.Verify;

public class VerifyDTO
{
    [JsonPropertyName("verified")]
    public bool Verified { get; set; }
    [JsonPropertyName("distance")]
    public float Distance { get; set; }
    [JsonPropertyName("threshold")]
    public float Threshold { get; set; }
    [JsonPropertyName("model")]
    public string? Model { get; set; }
    [JsonPropertyName("simularity_metric")]
    public string? SimilarityMetric { get; set; }
    [JsonPropertyName("facial_areas")]
    public FacialAreas? FacialAreas { get; set; }
}

public class FacialAreas
{
    [JsonPropertyName("img1")]
    public Region? Img1 { get; set; }
    [JsonPropertyName("img2")]
    public Region? Img2 { get; set; }
}