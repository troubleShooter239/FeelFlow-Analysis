using FeelFlowAnalysis.Models;
using FeelFlowAnalysis.Services;
using Microsoft.Extensions.Options;
using MongoDB.Driver;

namespace FeelFlowAnalysis;

public class Startup
{
    public static void ConfigureServices(WebApplicationBuilder builder)
        => builder.Services
            // Encryption service
            .Configure<EncryptionSettings>(builder.Configuration.GetSection(nameof(EncryptionSettings)))
            .AddSingleton<IEncryptionSettings>(sp => 
                sp.GetRequiredService<IOptions<EncryptionSettings>>().Value)
            .AddScoped<IEncryption, Encryption>()
            // Hashing service
            .Configure<HashingSettings>(builder.Configuration.GetSection(nameof(HashingSettings)))
            .AddSingleton<IHashingSettings>(sp => 
                sp.GetRequiredService<IOptions<HashingSettings>>().Value)
            .AddScoped<IHashing, Hashing>()
            // DB settings
            .Configure<DbSettings>(builder.Configuration.GetSection(nameof(DbSettings)))
            .AddSingleton<IDbSettings>(sp => 
                sp.GetRequiredService<IOptions<DbSettings>>().Value)
            .AddSingleton<IMongoClient>(sp =>
                new MongoClient(builder.Configuration.GetValue<string>("DbSettings:ConnectionString")))
            // User service
            .AddScoped<IUserService, UserService>()

            .AddRazorComponents().AddInteractiveServerComponents();       
}
