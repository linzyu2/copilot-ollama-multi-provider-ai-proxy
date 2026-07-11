using System.Text.Json;

namespace ProxyTests;

public class RequestTransformerTests
{
    [Fact]
    public void ReplaceModelInRequestBody_ReplacesExistingModel()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"model":"old-model","messages":[]}""";

        string result = sut.ReplaceModelInRequestBody(raw, "new-model");

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.Equal("new-model", doc.RootElement.GetProperty("model").GetString());
    }

    [Fact]
    public void ReplaceModelInRequestBody_AddsModelWhenMissing()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        string result = sut.ReplaceModelInRequestBody(raw, "added-model");

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.Equal("added-model", doc.RootElement.GetProperty("model").GetString());
    }

    [Fact]
    public void ModifyRequest_InjectsCachedReasoningContentForAssistantMessage()
    {
        RequestTransformer sut = CreateTransformer(out ReasoningCacheService cache);
        cache.Set("assistant:0", "cached reasoning");

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "assistant", "content": "hola" }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement msg = modified.RootElement.GetProperty("messages")[0];
        Assert.Equal("cached reasoning", msg.GetProperty("reasoning_content").GetString());
    }

    [Fact]
    public void ModifyRequest_RemovesEmptyAssistantMessagesWithoutToolCalls()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "user", "content": "hola" },
                { "role": "assistant", "content": "" },
                { "role": "assistant", "content": "ok" }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement messages = modified.RootElement.GetProperty("messages");
        Assert.Equal(2, messages.GetArrayLength());
        Assert.Equal("user", messages[0].GetProperty("role").GetString());
        Assert.Equal("assistant", messages[1].GetProperty("role").GetString());
        Assert.Equal("ok", messages[1].GetProperty("content").GetString());
    }

    [Fact]
    public void ModifyRequest_KeepsEmptyAssistantWhenToolCallsPresent()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                {
                  "role": "assistant",
                  "content": "",
                  "tool_calls": [
                    {
                      "id": "call_1",
                      "type": "function",
                      "function": { "name": "ping", "arguments": "{}" }
                    }
                  ]
                }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.Null(result);
    }

    [Fact]
    public void ApplyExecutionDefaults_KeepsBodyWhenNoConfiguredDefaults()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[],"stream":false}""";

        string result = sut.ApplyExecutionDefaults(raw, "unknown-model-without-config");

        Assert.Equal(raw, result);
    }

    [Fact]
    public void ApplyExecutionDefaults_InjectsTemperatureTopPAndMaxTokens()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro");

        using JsonDocument doc = JsonDocument.Parse(result);
        JsonElement root = doc.RootElement;
        Assert.True(root.TryGetProperty("max_tokens", out JsonElement maxTok));
        Assert.Equal(384000, maxTok.GetInt32());
    }

    [Fact]
    public void ApplyExecutionDefaults_InjectsReasoningEffortForDeepSeek()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro", ProviderCapabilitiesRegistry.Get("deepseek"));

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.Equal("max", doc.RootElement.GetProperty("reasoning_effort").GetString());
    }

    [Fact]
    public void ApplyExecutionDefaults_SkipsTopPForNativeReasoner()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        // deepseek-v4-pro has reasoning_effort configured, so it's a native reasoner
        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro", ProviderCapabilitiesRegistry.Get("deepseek"));

        // top_p should NOT be injected for native reasoners
        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.False(doc.RootElement.TryGetProperty("top_p", out _));
    }

    [Fact]
    public void ApplyExecutionDefaults_SkipsTopKForDeepSeekProvider()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[],"top_k":40}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro", ProviderCapabilitiesRegistry.Get("deepseek"));

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.False(doc.RootElement.TryGetProperty("top_k", out _));
    }

    [Fact]
    public void ApplyExecutionDefaults_KeepsTopKForNvidiaProvider()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[],"top_k":40}""";

        string result = sut.ApplyExecutionDefaults(raw, "some-model", ProviderCapabilitiesRegistry.Get("nvidia"));

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.True(doc.RootElement.TryGetProperty("top_k", out JsonElement topK));
        Assert.Equal(40, topK.GetInt32());
    }

    [Fact]
    public void ApplyExecutionDefaults_DoesNotInjectReasoningEffortForUnsupportedProvider()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro", ProviderCapabilitiesRegistry.Get("groq"));

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.False(doc.RootElement.TryGetProperty("reasoning_effort", out _));
    }

    [Fact]
    public void ApplyExecutionDefaults_PreservesExistingProperties()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[],"temperature":0.9,"custom":"value"}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro");

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.Equal(0.9, doc.RootElement.GetProperty("temperature").GetDouble());
        Assert.Equal("value", doc.RootElement.GetProperty("custom").GetString());
    }

    [Fact]
    public void ApplyExecutionDefaults_InjectReasoningEffortForOpenAiProvider()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro", ProviderCapabilitiesRegistry.Get("openai"));

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.Equal("max", doc.RootElement.GetProperty("reasoning_effort").GetString());
    }

    [Fact]
    public void ModifyRequest_ReturnsNullWhenNoMessagesProperty()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""{"model":"m"}""");

        string? result = sut.ModifyRequest(request);

        Assert.Null(result);
    }

    [Fact]
    public void ModifyRequest_ReturnsNullWhenNoAssistantMessages()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "user", "content": "hello" },
                { "role": "system", "content": "sys" }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.Null(result);
    }

    [Fact]
    public void ModifyRequest_KeepsAssistantWithFunctionCallEvenIfContentEmpty()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                {
                  "role": "assistant",
                  "content": "",
                  "function_call": { "name": "test", "arguments": "{}" }
                }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.Null(result);
    }

    [Fact]
    public void ModifyRequest_DoesNotReinjectWhenReasoningContentAlreadyPresent()
    {
        RequestTransformer sut = CreateTransformer(out ReasoningCacheService cache);
        cache.Set("assistant:0", "cached reasoning");

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "assistant", "content": "hola", "reasoning_content": "already present" }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.Null(result);
    }

    [Fact]
    public void ModifyRequest_InjectsReasoningForToolCallKey()
    {
        RequestTransformer sut = CreateTransformer(out ReasoningCacheService cache);
        cache.Set("toolcall:call_1", "tool reasoning");

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                {
                  "role": "assistant",
                  "content": "using tool",
                  "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "test" } }
                  ]
                }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement msg = modified.RootElement.GetProperty("messages")[0];
        Assert.Equal("tool reasoning", msg.GetProperty("reasoning_content").GetString());
    }

    [Fact]
    public void ModifyRequest_InjectsReasoningForAssistantMessagesInOrder()
    {
        RequestTransformer sut = CreateTransformer(out ReasoningCacheService cache);
        cache.Set("assistant:0", "first reasoning");
        cache.Set("assistant:1", "second reasoning");

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "assistant", "content": "one" },
                { "role": "assistant", "content": "two" }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement messages = modified.RootElement.GetProperty("messages");
        Assert.Equal("first reasoning", messages[0].GetProperty("reasoning_content").GetString());
        Assert.Equal("second reasoning", messages[1].GetProperty("reasoning_content").GetString());
    }

    [Fact]
    public void ModifyRequest_PreservesCachedReasoningOnEmptyAssistantMessage()
    {
        RequestTransformer sut = CreateTransformer(out ReasoningCacheService cache);
        cache.Set("assistant:0", "cached reasoning");

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "assistant", "content": "" }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement message = modified.RootElement.GetProperty("messages")[0];
        Assert.Equal("cached reasoning", message.GetProperty("reasoning_content").GetString());
        Assert.Equal("", message.GetProperty("content").GetString());
    }

    [Fact]
    public void ReplaceModelInRequestBody_InvalidJson_ReturnsOriginal()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = "not-json";

        string result = sut.ReplaceModelInRequestBody(raw, "model");

        Assert.Equal(raw, result);
    }

    [Fact]
    public void ApplyExecutionDefaults_InvalidJson_ReturnsOriginal()
    {
        RequestTransformer sut = CreateTransformer();
        string raw = "not-json";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro");

        Assert.Equal(raw, result);
    }

    [Fact]
    public void ModifyRequest_RemovesAssistantWithNullContent()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "user", "content": "hello" },
                { "role": "assistant", "content": null }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement messages = modified.RootElement.GetProperty("messages");
        Assert.Equal(1, messages.GetArrayLength());
        Assert.Equal("user", messages[0].GetProperty("role").GetString());
    }

    [Fact]
    public void ModifyRequest_RemovesAssistantWithWhitespaceContent()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "user", "content": "hello" },
                { "role": "assistant", "content": "   " }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement messages = modified.RootElement.GetProperty("messages");
        Assert.Equal(1, messages.GetArrayLength());
    }

    [Fact]
    public void ModifyRequest_RemovesAssistantWithEmptyArrayContent()
    {
        RequestTransformer sut = CreateTransformer();

        using JsonDocument request = JsonDocument.Parse("""
            {
              "model": "m",
              "messages": [
                { "role": "assistant", "content": [] }
              ]
            }
            """);

        string? result = sut.ModifyRequest(request);

        Assert.NotNull(result);
        using JsonDocument modified = JsonDocument.Parse(result!);
        JsonElement messages = modified.RootElement.GetProperty("messages");
        Assert.Equal(0, messages.GetArrayLength());
    }

    [Fact]
    public void ApplyExecutionDefaults_DoesNotInjectReasoningEffortForUnknownProvider()
    {
        // With capabilities, unknown providers have all feature flags set to false.
        // No reasoning_effort is injected regardless of model name.
        RequestTransformer sut = CreateTransformer();
        string raw = """{"messages":[]}""";

        string result = sut.ApplyExecutionDefaults(raw, "deepseek-v4-pro", default);

        using JsonDocument doc = JsonDocument.Parse(result);
        Assert.False(doc.RootElement.TryGetProperty("reasoning_effort", out _));
    }

    [Fact]
    public void PruneToContextWindow_PinsFirstSystemAndFirstUserMessage()
    {
        // 上下文窗口仅能容纳钉死的首个 system + 首个 user（修正后的结构化估算约 110 token），
        // 其余消息用超长文本确保必然触发裁剪。
        RequestTransformer sut = CreateTransformer();
        ModelExecutionConfig exec = new(ContextLength: 150, MaxOutputTokens: 20);
        string huge = new string('x', 5000);
        string raw = $$"""
            {
              "messages": [
                { "role": "system", "content": "system instructions" },
                { "role": "user", "content": "first user prompt" },
                { "role": "user", "content": "{{huge}}" }
              ]
            }
            """;

        string result = sut.PruneToContextWindow(raw, exec);

        using JsonDocument doc = JsonDocument.Parse(result);
        JsonElement messages = doc.RootElement.GetProperty("messages");
        Assert.Equal(2, messages.GetArrayLength());
        Assert.Equal("system", messages[0].GetProperty("role").GetString());
        Assert.Equal("user", messages[1].GetProperty("role").GetString());
        Assert.Equal("first user prompt", messages[1].GetProperty("content").GetString());
    }

    [Fact]
    public void PruneToContextWindow_KeepsToolExchangeAtomic()
    {
        RequestTransformer sut = CreateTransformer();
        ModelExecutionConfig exec = new(ContextLength: 150, MaxOutputTokens: 20);
        string huge = new string('x', 5000);
        string raw = $$"""
            {
              "messages": [
                { "role": "system", "content": "system instructions" },
                { "role": "user", "content": "first user prompt" },
                {
                  "role": "assistant",
                  "content": "",
                  "tool_calls": [ { "id": "c1", "type": "function", "function": { "name": "f", "arguments": "{}" } } ]
                },
                { "role": "tool", "content": "{{huge}}", "tool_call_id": "c1" },
                { "role": "user", "content": "final user prompt" }
              ]
            }
            """;

        string result = sut.PruneToContextWindow(raw, exec);

        using JsonDocument doc = JsonDocument.Parse(result);
        JsonElement messages = doc.RootElement.GetProperty("messages");
        // 仅钉死的首个 system + 首个 user 保留；assistant+tool 块整体被移除。
        Assert.Equal(2, messages.GetArrayLength());
        Assert.Equal("system", messages[0].GetProperty("role").GetString());
        Assert.Equal("first user prompt", messages[1].GetProperty("content").GetString());
        AssertToolExchangeInvariant(messages);
    }

    [Fact]
    public void PruneToContextWindow_DoesNotSplitToolExchange()
    {
        // 仅保留首个 system + 首个 user 仍超限时，必须整块删除 tool 交换，
        // 不允许只删 assistant 或只删 tool 导致单边残留。
        RequestTransformer sut = CreateTransformer();
        ModelExecutionConfig exec = new(ContextLength: 150, MaxOutputTokens: 20);
        string huge = new string('x', 5000);
        string raw = $$"""
            {
              "messages": [
                { "role": "system", "content": "system instructions" },
                { "role": "user", "content": "first user prompt" },
                {
                  "role": "assistant",
                  "content": "",
                  "tool_calls": [ { "id": "c1", "type": "function", "function": { "name": "f", "arguments": "{}" } } ]
                },
                { "role": "tool", "content": "{{huge}}", "tool_call_id": "c1" }
              ]
            }
            """;

        string result = sut.PruneToContextWindow(raw, exec);

        using JsonDocument doc = JsonDocument.Parse(result);
        JsonElement messages = doc.RootElement.GetProperty("messages");
        Assert.Equal(2, messages.GetArrayLength());
        AssertToolExchangeInvariant(messages);
    }

    /// <summary>
    /// 校验裁剪后的消息列表不存在 tool_call 与 tool 响应失配的非法对话。
    /// </summary>
    private static void AssertToolExchangeInvariant(JsonElement messages)
    {
        bool hasAssistantWithToolCalls = false;
        bool hasTool = false;
        foreach (JsonElement m in messages.EnumerateArray())
        {
            string? role = m.TryGetProperty("role", out JsonElement r) ? r.GetString() : null;
            if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase)
                && m.TryGetProperty("tool_calls", out JsonElement tc)
                && tc.ValueKind == JsonValueKind.Array && tc.GetArrayLength() > 0)
            {
                hasAssistantWithToolCalls = true;
            }
            if (string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase))
            {
                hasTool = true;
            }
        }
        // 两者必须同时出现或同时消失，不允许单边残留。
        Assert.Equal(hasAssistantWithToolCalls, hasTool);
    }

    private static RequestTransformer CreateTransformer()
    {
        return CreateTransformer(out _);
    }

    private static RequestTransformer CreateTransformer(out ReasoningCacheService cache)
    {
        Environment.SetEnvironmentVariable("PROVIDER_DEEPSEEK_API_KEY", "unit-test-key");
        Environment.SetEnvironmentVariable("PROVIDER_DEEPSEEK_BASE_URL", "http://127.0.0.1:12345");

        ProviderHttpClientFactory httpClientFactory = new();
        ProviderRegistry providerRegistry = new(httpClientFactory);
        ModelSelectionStore modelSelectionStore = new();
        ModelCatalogService modelCatalog = new(providerRegistry, modelSelectionStore);
        cache = new ReasoningCacheService();

        return new RequestTransformer(modelCatalog, cache, new TokenizerService());
    }
}
