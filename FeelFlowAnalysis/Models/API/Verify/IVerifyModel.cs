namespace FeelFlowAnalysis.Models.API.Verify;

public interface IVerifyModel
{
    string B64img1 { get; set; }
    string B64img2 { get; set; }
    string RModelName { get; set; }
    string DistanceMetric { get; set; }
}
