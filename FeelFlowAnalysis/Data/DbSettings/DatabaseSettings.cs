namespace FeelFlowAnalysis.Data.DbSettings;

public class DatabaseSettings : IDatabaseSettings
{
    public string ConnectionString { get; set; } = string.Empty;
    public string DatabaseName { get; set; } = string.Empty;
    public string UsersCollectionName { get; set; } = string.Empty;
    public string ProductsCollectionName { get; set; } = string.Empty;
}
