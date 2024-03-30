using FeelFlowAnalysis.Services.Implementations;
using FeelFlowAnalysis.Services.Interfaces;
using Microsoft.AspNetCore.Authentication.Cookies;
using MongoDB.Driver;

namespace FeelFlowAnalysis;

public sealed class Startup
{
    public static void ConfigureServices(WebApplicationBuilder builder)
    {
        IConfigurationSection section = builder.Configuration.GetSection(nameof(Settings));
        Settings settings = section.Get<Settings>()!;
        builder.Services
            // Configuring settings
            .Configure<Settings>(section)
            // Encryption service
            .AddScoped<IEncryptionService, EncryptionService>()
            // Hashing service
            .AddScoped<IHashingService, HashingService>()
            // DB clinet
            .AddSingleton<IMongoClient>(sp => new MongoClient(settings.Database.ConnectionString))
            // User service
            .AddScoped<IUserService, UserService>()
            // Authorization service
            .AddAuthorization()
            .AddCascadingAuthenticationState()
            // Cookie auth service
            .AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
            .AddCookie(co => 
                {
                    co.Cookie.Name = "auth_token";
                    co.LoginPath = "/login";
                    co.Cookie.MaxAge = TimeSpan.FromMinutes(30);
                    co.AccessDeniedPath = "/access-denied"; // TODO: Add access-denied or delete this line
                }); 
            
        builder.Services
            .AddRazorComponents()
            .AddInteractiveServerComponents();
        
        builder.Services.AddHttpClient("", c => c.BaseAddress= new Uri(settings.Api.BaseUrl));
    }       
}
