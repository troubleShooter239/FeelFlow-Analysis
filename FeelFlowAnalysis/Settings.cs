namespace FeelFlowAnalysis;

public sealed class Settings
{
    public required ApiSettings Api { get; set; }
    public required DatabaseSettings Database { get; set; }
    public required EncryptionSettings Encryption { get; set; }
    public required HashingSettings Hashing { get; set; }

    public sealed class ApiSettings
    {
        public required string BaseUrl { get; set; }
        public required string Analyze { get; set; }
        public int MaxFileSize { get; set; }
        public required string Metadata { get; set; }
        public required string Verify { get; set; }
        public int VerifyMaxFiles { get; set; }
    }
    public sealed class DatabaseSettings
    {
        public required string ConnectionString { get; set; }
        public required string Name { get; set; }
        public required string UsersCollectionName { get; set; }
    }
    public sealed class EncryptionSettings
    {
        public required string EncryptionKey { get; set; }
        public required string InitializationVector { get; set; }
    }
    public sealed class HashingSettings
    {
        public int SaltSize { get; set; }
        public int HashSize { get; set; }
        public int Iterations { get; set; }
    }
}