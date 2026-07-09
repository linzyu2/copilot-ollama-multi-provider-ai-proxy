using System.Text.Json;

namespace ProxyTests;

public class OllamaResponseBuilderTests
{
    private static OllamaResponseBuilder CreateBuilder()
    {
        Environment.SetEnvironmentVariable("PROVIDER_DEEPSEEK_API_KEY", "unit-test-key");
        Environment.SetEnvironmentVariable("PROVIDER_DEEPSEEK_BASE_URL", "http://127.0.0.1:12345");

        ProviderHttpClientFactory httpClientFactory = new();
        ProviderRegistry providerRegistry = new(httpClientFactory);
        ModelSelectionStore modelSelectionStore = new();
        ModelCatalogService modelCatalog = new(providerRegistry, modelSelectionStore);

        return new OllamaResponseBuilder(modelCatalog);
    }

    [Fact]
    public void BuildOllamaShowResponse_ReturnsModelName()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("modelfile"));
        Assert.Contains("deepseek-v4-pro", result["modelfile"]?.ToString());
    }

    [Fact]
    public void BuildOllamaShowResponse_HasModelInfo()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("model_info"));
        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        Assert.True(modelInfo.ContainsKey("context_length"));
        Assert.True(modelInfo.ContainsKey("max_output_tokens"));
    }

    [Fact]
    public void BuildOllamaShowResponse_HasContextLength()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        Assert.True(modelInfo.ContainsKey("context_length"));
        int ctx = Convert.ToInt32(modelInfo["context_length"]);
        Assert.True(ctx > 0);
    }

    [Fact]
    public void BuildOllamaShowResponse_HasMaxOutputTokens()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        Assert.True(modelInfo.ContainsKey("max_output_tokens"));
        int maxOut = Convert.ToInt32(modelInfo["max_output_tokens"]);
        Assert.True(maxOut > 0);
    }

    [Fact]
    public void BuildOllamaShowResponse_HasCapabilities()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("capabilities"));
        string[] caps = (string[])result["capabilities"]!;
        Assert.Contains("completion", caps);
        Assert.Contains("tools", caps);
    }

    [Fact]
    public void BuildOllamaShowResponse_HasRecommendedParameters()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        Assert.True(modelInfo.ContainsKey("proxy.recommended_parameters"));
        Dictionary<string, object?> recParams = (Dictionary<string, object?>)modelInfo["proxy.recommended_parameters"]!;
        Assert.NotEmpty(recParams);
    }

    [Fact]
    public void BuildOllamaShowResponse_HasDetails()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("details"));
        Dictionary<string, object?> details = (Dictionary<string, object?>)result["details"]!;
        Assert.True(details.ContainsKey("family"));
        Assert.True(details.ContainsKey("format"));
    }

    [Fact]
    public void BuildOllamaShowResponse_HasParametersString()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("parameters"));
        string? paramStr = result["parameters"]?.ToString();
        Assert.False(string.IsNullOrWhiteSpace(paramStr));
        Assert.Contains("num_ctx", paramStr);
        Assert.Contains("num_predict", paramStr);
    }

    [Fact]
    public void BuildOllamaShowResponse_ModelInfo_HasContextLength()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        int ctx = Convert.ToInt32(modelInfo["general.context_length"]);
        Assert.True(ctx > 0);
    }

    [Fact]
    public void BuildOllamaShowResponse_ModelInfo_HasSupportsTools()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        Assert.True(Convert.ToBoolean(modelInfo["supports_tools"]));
        Assert.True(Convert.ToBoolean(modelInfo["supports_tool_calls"]));
    }

    [Fact]
    public void BuildOllamaShowResponse_SupportsVisionFlag()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        Assert.True(modelInfo.ContainsKey("supports_vision"));
        Assert.True(modelInfo.ContainsKey("supports_images"));
    }

    [Fact]
    public void BuildOllamaShowResponse_DifferentModelsHaveDifferentProfiles()
    {
        OllamaResponseBuilder builder = CreateBuilder();

        Dictionary<string, object?> result1 = builder.BuildOllamaShowResponse("deepseek-v4-pro");
        Dictionary<string, object?> result2 = builder.BuildOllamaShowResponse("gpt-5");

        int ctx1 = Convert.ToInt32(((Dictionary<string, object?>)result1["model_info"]!)["context_length"]);
        int ctx2 = Convert.ToInt32(((Dictionary<string, object?>)result2["model_info"]!)["context_length"]);

        Assert.NotEqual(ctx1, ctx2);
    }

    [Fact]
    public void BuildOllamaShowResponse_HasModifiedAt()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("modified_at"));
        string? modifiedAt = result["modified_at"]?.ToString();
        Assert.False(string.IsNullOrWhiteSpace(modifiedAt));
    }

    [Fact]
    public void BuildOllamaShowResponse_HasLicense()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Assert.True(result.ContainsKey("license"));
        Assert.Equal("NIM API", result["license"]);
    }

    [Fact]
    public void BuildOllamaShowResponse_DetailsHasContextLength()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("deepseek-v4-pro");

        Dictionary<string, object?> details = (Dictionary<string, object?>)result["details"]!;
        Assert.True(details.ContainsKey("context_length"));
        int ctx = Convert.ToInt32(details["context_length"]);
        Assert.Equal(
            Convert.ToInt32(((Dictionary<string, object?>)result["model_info"]!)["context_length"]),
            ctx);
    }

    [Fact]
    public void BuildOllamaShowResponse_SupportsReasoningMapsToCapabilities()
    {
        OllamaResponseBuilder builder = CreateBuilder();
        Dictionary<string, object?> result = builder.BuildOllamaShowResponse("kimi-k2.7-code");

        string[] caps = (string[])result["capabilities"]!;
        Assert.Contains("reason", caps);

        Dictionary<string, object?> modelInfo = (Dictionary<string, object?>)result["model_info"]!;
        string[] generalCaps = (string[])modelInfo["general.capabilities"]!;
        Assert.Contains("reason", generalCaps);
    }
}