namespace FeelFlowAnalysis;

public sealed class Settings
{
    public required ApiSettings Api { get; set; }
    public required DatabaseSettings Database { get; set; }
    public required EncryptionSettings Encryption { get; set; }
    public required HashingSettings Hashing { get; set; }

    public sealed class ApiSettings
    {
        public string BaseUrl { get; set; } = string.Empty;
        public string Analyze { get; set; } = string.Empty;
        public string Metadata { get; set; } = string.Empty;
        public string Verify { get; set; } = string.Empty;
    }
    public sealed class DatabaseSettings
    {
        public string ConnectionString { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string UsersCollectionName { get; set; } = string.Empty;
    }
    public sealed class EncryptionSettings
    {
        public string EncryptionKey { get; set; } = string.Empty;
        public string InitializationVector { get; set; } = string.Empty;
    }
    public sealed class HashingSettings
    {
        public int SaltSize { get; set; }
        public int HashSize { get; set; }
        public int Iterations { get; set; }
    }
}