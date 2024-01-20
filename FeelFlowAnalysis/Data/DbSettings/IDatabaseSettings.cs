namespace FeelFlowAnalysis.Data.DbSettings;

public interface IDatabaseSettings
{
    string ConnectionString { get; set; }
    string DatabaseName { get; set; }
    string UsersCollectionName { get; set; }
    string ProductsCollectionName { get; set; }
}
