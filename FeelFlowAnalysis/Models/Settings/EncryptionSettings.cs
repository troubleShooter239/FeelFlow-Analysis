namespace FeelFlowAnalysis.Models.Settings;

public class EncryptionSettings : IEncryptionSettings
{
    public string EncryptionKey { get; set; } = string.Empty;
    public string InitializationVector { get; set; } = string.Empty;
}
