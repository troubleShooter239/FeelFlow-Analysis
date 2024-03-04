namespace FeelFlowAnalysis.Models;

public class HashingSettings : IHashingSettings
{
    public int SaltSize { get; set; }
    public int HashSize { get; set; }
    public int Iterations { get; set; }
}
