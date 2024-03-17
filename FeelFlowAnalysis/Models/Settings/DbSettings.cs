namespace FeelFlowAnalysis.Models.Settings;

public class DbSettings : IDbSettings
{
    public string ConnectionString { get; set; } = string.Empty;
    public string DatabaseName { get; set; } = string.Empty;
    public string UsersCollectionName { get; set; } = string.Empty;
}

