namespace FeelFlowAnalysis.Models;

public interface IEncryptionSettings
{
    string EncryptionKey { get; set; }
    string InitializationVector { get; set; }
}
