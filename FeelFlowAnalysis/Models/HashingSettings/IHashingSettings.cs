namespace FeelFlowAnalysis.Models;

public interface IHashingSettings
{
    int SaltSize { get; set; }
    int HashSize { get; set; }
    int Iterations { get; set; }
}
