namespace FeelFlowAnalysis.Models.DTO.Verify;

public class VerifyDTO
{
    public bool Verified { get; set; }
    public float Distance { get; set; }
    public float Threshold { get; set; }
    public string? Model { get; set; }
    public string? SimilarityMetric { get; set; }
    public Dictionary<string, Region>? FacialAreas { get; set; }
}
