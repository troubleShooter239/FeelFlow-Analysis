using FeelFlowAnalysis.Models.Settings;
using FeelFlowAnalysis.Services.Implementations;
using FeelFlowAnalysis.Services.Interfaces;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.Extensions.Options;
using MongoDB.Driver;

namespace FeelFlowAnalysis;

public class Startup
{
    public static void ConfigureServices(WebApplicationBuilder builder)
    {
        builder.Services
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
            // Authorization service
            .AddAuthorization()
            .AddCascadingAuthenticationState()
            
            .AddRazorComponents()
            .AddInteractiveServerComponents();
            
            // Cookie auth service
        builder.Services
            .AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
            .AddCookie(options => 
            {
                options.Cookie.Name = "auth_token";
                options.LoginPath = "/login";
                options.Cookie.MaxAge = TimeSpan.FromMinutes(30);
                options.AccessDeniedPath = "/access-denied";
            });   
    }       
}
