namespace FeelFlowAnalysis.Models.Settings;

public interface IEncryptionSettings
{
    string EncryptionKey { get; set; }
    string InitializationVector { get; set; }
}
