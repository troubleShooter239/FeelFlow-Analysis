namespace FeelFlowAnalysis.Models;

public interface IDbSettings
{
    string ConnectionString { get; set; }
    string DatabaseName { get; set; }
    string UsersCollectionName { get; set; }
    string ProductsCollectionName { get; set; }
}

