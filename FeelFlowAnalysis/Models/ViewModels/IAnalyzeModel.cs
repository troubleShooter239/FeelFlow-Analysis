namespace FeelFlowAnalysis.Models.ViewModels;

public interface IAnalyzeModel
{
    string B64img { get; set; }
    Dictionary<string, bool> Actions { get; set; }
}
