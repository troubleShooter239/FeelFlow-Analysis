namespace FeelFlowAnalysis.Models.DTO.Verify;

public class VerifyDTO
{
    public bool Verified { get; set; }
    public float Distance { get; set; }
    public float Threshold { get; set; }
    public required string Model { get; set; }
    public required string SimilarityMetric { get; set; }
    public required Dictionary<string, Region> FacialAreas { get; set; }
}
