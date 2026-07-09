using System.Text.Json;

namespace ProxyTests;

// Share the "Proxy" collection so ProxyFixture's environment-variable setup
// runs before any of these tests, and the registry constructor can find a
// configured PROVIDER_DEEPSEEK_API_KEY.
[Collection("Proxy")]
public class ProviderRegistryTests
{
    [Fact]
    public void ResolveProvider_WithNullModel_ReturnsDefaultProvider()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        ProviderInfo result = registry.ResolveProvider(null);

        
        Assert.Equal("deepseek", result.Name);
    }

    [Fact]
    public void ResolveProvider_WithEmptyModel_ReturnsDefaultProvider()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        ProviderInfo result = registry.ResolveProvider("");

        
        Assert.Equal("deepseek", result.Name);
    }

    [Fact]
    public void ResolveModel_WithNullModel_ReturnsDefaultModel()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        string result = registry.ResolveModel(null);

        Assert.Equal("deepseek-v4-flash", result);
    }

    [Fact]
    public void ResolveModel_WithEmptyModel_ReturnsDefaultModel()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        string result = registry.ResolveModel("");

        Assert.Equal("deepseek-v4-flash", result);
    }

    [Fact]
    public void ResolveUpstreamModel_WithNullModel_ReturnsDefaultModel()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        string result = registry.ResolveUpstreamModel(null);

        Assert.Equal("deepseek-v4-flash", result);
    }

    [Fact]
    public void DefaultModel_IsDeepSeekV4Flash()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        Assert.Equal("deepseek-v4-flash", registry.DefaultModel);
    }

    [Fact]
    public void UpdateModelMappings_UpdatesModelToProvider()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        Dictionary<string, ProviderInfo> newMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["custom-model"] = new ProviderInfo("groq", "key", "http://localhost", new System.Net.Http.HttpClient(), ProviderCapabilitiesRegistry.Get("groq"))
        };
        Dictionary<string, string> newUpstream = new(StringComparer.OrdinalIgnoreCase)
        {
            ["custom-model"] = "llama-3.3-70b-versatile"
        };

        registry.UpdateModelMappings(newMap, newUpstream);

        ProviderInfo result = registry.ResolveProvider("custom-model");
        
        Assert.Equal("groq", result.Name);
    }

    [Fact]
    public void ResolveModel_WithDisplayLabelAndTag_ResolvesMappedModel()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        ProviderInfo deepseek = registry.Providers.First(p => p.Name == "deepseek");
        Dictionary<string, ProviderInfo> newMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["deepseek-v4-flash"] = deepseek,
            ["deepseek-v4-flash@deepseek"] = deepseek
        };
        Dictionary<string, string> newUpstream = new(StringComparer.OrdinalIgnoreCase)
        {
            ["deepseek-v4-flash"] = "deepseek-v4-flash",
            ["deepseek-v4-flash@deepseek"] = "deepseek-v4-flash"
        };

        registry.UpdateModelMappings(newMap, newUpstream);

        string result = registry.ResolveModel("DEEPSEEK - deepseek-v4-flash:latest");

        Assert.Equal("deepseek-v4-flash@deepseek", result);
    }

    [Fact]
    public void ResolveModel_WithDisplayLabelAndTag_DifferentProvider_RoutesCorrectly()
    {
        // When the display prefix names a different provider, the resolved
        // model should include the @provider qualifier so it doesn't fall
        // back to the highest-priority provider for the bare model name.
        Environment.SetEnvironmentVariable("PROVIDER_OPENROUTER_API_KEY", "test-key");
        Environment.SetEnvironmentVariable("PROVIDER_OPENROUTER_BASE_URL", "http://openrouter.test");
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        ProviderInfo openrouter = registry.Providers.First(p => p.Name == "openrouter");
        ProviderInfo deepseek = registry.Providers.First(p => p.Name == "deepseek");
        Dictionary<string, ProviderInfo> newMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["deepseek-v4-pro"] = deepseek,
            ["deepseek-v4-pro@openrouter"] = openrouter
        };
        Dictionary<string, string> newUpstream = new(StringComparer.OrdinalIgnoreCase)
        {
            ["deepseek-v4-pro"] = "deepseek-v4-pro",
            ["deepseek-v4-pro@openrouter"] = "deepseek-v4-pro"
        };

        registry.UpdateModelMappings(newMap, newUpstream);

        string result = registry.ResolveModel("OPENROUTER - deepseek-v4-pro:latest");

        Assert.Equal("deepseek-v4-pro@openrouter", result);
    }

    [Fact]
    public void ResolveProvider_WithDisplayLabelAndTag_ResolvesMappedProvider()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        ProviderInfo deepseek = registry.Providers.First(p => p.Name == "deepseek");
        Dictionary<string, ProviderInfo> newMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["deepseek-v4-flash"] = deepseek,
            ["deepseek-v4-flash@deepseek"] = deepseek
        };
        Dictionary<string, string> newUpstream = new(StringComparer.OrdinalIgnoreCase)
        {
            ["deepseek-v4-flash"] = "deepseek-v4-flash",
            ["deepseek-v4-flash@deepseek"] = "deepseek-v4-flash"
        };

        registry.UpdateModelMappings(newMap, newUpstream);

        ProviderInfo result = registry.ResolveProvider("DEEPSEEK - deepseek-v4-flash:latest");

        Assert.Equal("deepseek", result.Name);
    }

    [Fact]
    public void ResolveCandidates_WithNullModel_ReturnsDefaultProvider()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        var candidates = registry.ResolveCandidates(null);

        Assert.Single(candidates);
        Assert.Equal("deepseek", candidates[0].Provider.Name);
    }

    [Fact]
    public void Providers_AtLeastOneProviderExists()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        Assert.NotEmpty(registry.Providers);
    }

    [Fact]
    public void ModelToProvider_IsNotNull()
    {
        ProviderHttpClientFactory factory = new();
        ProviderRegistry registry = new(factory);

        Assert.NotNull(registry.ModelToProvider);
    }
}