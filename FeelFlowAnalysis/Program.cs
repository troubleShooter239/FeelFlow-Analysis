using FeelFlowAnalysis;
using FeelFlowAnalysis.Components;

var builder = WebApplication.CreateBuilder(args);

Startup.ConfigureServices(builder);

var app = builder.Build();

if (!app.Environment.IsDevelopment())
    app.UseExceptionHandler("/Error", createScopeForErrors: true)
        .UseHsts();

app.UseHttpsRedirection()
    .UseStaticFiles()
    .UseAntiforgery()
    // Add auth to application
    .UseAuthentication()
    .UseAuthorization();

app.MapRazorComponents<App>().AddInteractiveServerRenderMode();

app.Run();
