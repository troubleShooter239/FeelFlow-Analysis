using FeelFlowAnalysis.Models;
using FeelFlowAnalysis.Services;
using Microsoft.Extensions.Options;

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

            .AddRazorComponents().AddInteractiveServerComponents();       
}
