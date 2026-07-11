using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

internal static class OpenAiEndpoints
{
    internal static IEndpointRouteBuilder MapOpenAiEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/v1/models", (HttpContext ctx, ModelCatalogService modelCatalog, ProviderRegistry providerRegistry) =>
        {
            _ = modelCatalog.RefreshAvailableModelsIfNeeded(ctx.RequestAborted);
            List<(string Provider, string Model)> allModels = BuildModelList(modelCatalog, providerRegistry);

            return Results.Json(new
            {
                @object = "list",
                data = allModels.Select(m => new
                {
                    id = m.Model,
                    @object = "model",
                    created = 1700000000,
                    owned_by = m.Provider
                }).ToArray()
            }, JsonDefaults.SnakeCase);
        });

        app.MapPost("/v1/chat/completions", async (
            HttpContext ctx,
            ProviderRegistry providerRegistry,
            RequestTransformer requestTransformer,
            ModelCatalogService modelCatalog,
            ChatStreamingService chatStreaming,
            ReasoningCacheService reasoningCache,
            ModelConcurrencyLimiter concurrencyLimiter,
            TokenizerService tokenizer) =>
        {
            CancellationToken ct = ctx.RequestAborted;

            using StreamReader bodyReader = new(ctx.Request.Body, Encoding.UTF8, false, 1024);
            string rawBody = await bodyReader.ReadToEndAsync(ct);

            using JsonDocument doc = JsonDocument.Parse(rawBody);
            JsonElement root = doc.RootElement;
            bool isStream = root.TryGetProperty("stream", out JsonElement sp) && sp.GetBoolean();

            string reqModel = root.TryGetProperty("model", out JsonElement rm) && rm.ValueKind == JsonValueKind.String
                ? rm.GetString()! : providerRegistry.DefaultModel;

            // 单一生命周期标识 id = 会话根 + 时分秒。会话根来自 VS 自带的会话头
            // （X-Copilot-Chat-Conversation-Id / X-Github-Session-Id / traceparent），
            // 没有时回退到随机串；时分秒保证同一会话的多轮请求也能彼此区分。
            string sessionRoot = ExtractSessionId(ctx);
            string id = $"{(string.IsNullOrEmpty(sessionRoot) ? Guid.NewGuid().ToString("N")[..8] : sessionRoot)}_{DateTime.Now:HHmmss}";
            Stopwatch sw = Stopwatch.StartNew();
            int inputTokens = tokenizer.CountTokens(rawBody);
            Console.WriteLine($"\n[REQ] model=\"{reqModel}\" stream={isStream} bytes={rawBody.Length} intok={inputTokens} id={id}");

            string effectiveModel = providerRegistry.ResolveModel(reqModel);
            IReadOnlyList<(ProviderInfo Provider, string UpstreamModel)> candidates =
                ResolveCandidates(reqModel, effectiveModel, providerRegistry);
            AddRoutingHeaders(ctx, reqModel, effectiveModel, candidates);

            string? modifiedRequest = requestTransformer.ModifyRequest(doc);
            ModelExecutionConfig executionConfig = modelCatalog.GetExecutionConfigForModel(effectiveModel);

            using CancellationTokenSource? timeoutCts = modelCatalog.CreateModelTimeoutCts(effectiveModel, ct);
            CancellationToken requestCt = timeoutCts?.Token ?? ct;
            await using ModelConcurrencyLimiter.Lease concurrencyLease = await concurrencyLimiter.AcquireAsync(
                effectiveModel,
                executionConfig.MaxConcurrency,
                requestCt);

            if (!isStream)
            {
                await HandleNonStreamingCompletionAsync(
                    ctx, candidates, modifiedRequest ?? rawBody, effectiveModel, requestTransformer,
                    reasoningCache, sessionRoot, requestCt, ct, id, sw);
                return;
            }

            await HandleStreamingCompletionAsync(
                ctx, candidates[0], modifiedRequest ?? rawBody, effectiveModel, requestTransformer,
                chatStreaming, sessionRoot, requestCt, ct, id, sw);
        });

        return app;
    }

    private static List<(string Provider, string Model)> BuildModelList(
        ModelCatalogService modelCatalog,
        ProviderRegistry providerRegistry)
    {
        List<(string Provider, string Model)> models = [];
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        List<(string Provider, string Model)> bareCandidates = [];

        foreach (string modelId in modelCatalog.AvailableModels)
        {
            if (string.IsNullOrWhiteSpace(modelId) || !seen.Add(modelId))
                continue;

            string providerName = GetModelProviderName(modelId, providerRegistry);
            if (modelId.Contains('@'))
                models.Add((providerName, modelId));
            else
                bareCandidates.Add((providerName, modelId));
        }

        foreach ((string provider, string model) in bareCandidates)
        {
            if (!seen.Contains($"{model}@{provider}"))
                models.Add((provider, model));
        }

        foreach (KeyValuePair<string, ProviderInfo> entry in providerRegistry.ModelToProvider)
        {
            if (!seen.Add(entry.Key))
                continue;

            if (!entry.Key.Contains('@') && seen.Contains($"{entry.Key}@{entry.Value.Name}"))
                continue;

            models.Add((entry.Value.Name, entry.Key));
        }

        return models.OrderBy(model => model.Provider, StringComparer.OrdinalIgnoreCase)
            .ThenBy(model => model.Model, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static string GetModelProviderName(string modelId, ProviderRegistry providerRegistry)
    {
        int at = modelId.IndexOf('@');
        if (at > 0 && at < modelId.Length - 1)
            return modelId[(at + 1)..];

        return providerRegistry.ModelToProvider.TryGetValue(modelId, out ProviderInfo provider)
            ? provider.Name
            : "unknown";
    }

    private static IReadOnlyList<(ProviderInfo Provider, string UpstreamModel)> ResolveCandidates(
        string requestedModel,
        string effectiveModel,
        ProviderRegistry providerRegistry)
    {
        ProviderInfo? requestedProvider = ExtractProviderHint(requestedModel, providerRegistry);
        if (requestedProvider is { } pinnedProvider)
        {
            string upstreamModel = providerRegistry.ResolveUpstreamModel(effectiveModel);
            return [(pinnedProvider, upstreamModel)];
        }

        return providerRegistry.ResolveCandidates(effectiveModel);
    }

    private static void AddRoutingHeaders(
        HttpContext ctx,
        string requestedModel,
        string effectiveModel,
        IReadOnlyList<(ProviderInfo Provider, string UpstreamModel)> candidates)
    {
        ctx.Response.Headers["X-Proxy-Requested-Model"] = requestedModel;
        ctx.Response.Headers["X-Proxy-Resolved-Model"] = effectiveModel;
        ctx.Response.Headers["X-Proxy-Candidate-Count"] = candidates.Count.ToString();

        if (candidates.Count > 0)
        {
            ctx.Response.Headers["X-Proxy-Primary-Provider"] = candidates[0].Provider.Name;
            ctx.Response.Headers["X-Proxy-Primary-Upstream"] = candidates[0].UpstreamModel;
        }
    }

    private static async Task HandleNonStreamingCompletionAsync(
        HttpContext ctx,
        IReadOnlyList<(ProviderInfo Provider, string UpstreamModel)> candidates,
        string requestBody,
        string effectiveModel,
        RequestTransformer requestTransformer,
        ReasoningCacheService reasoningCache,
        string sessionRoot,
        CancellationToken requestCt,
        CancellationToken clientCt,
        string id,
        Stopwatch sw)
    {
        HttpResponseMessage? lastResponse = null;
        string? lastBody = null;
        try
        {
            foreach ((ProviderInfo provider, string upstreamModel) in candidates)
            {
                string candidateBody = PrepareUpstreamRequestBody(requestBody, upstreamModel, effectiveModel, provider, requestTransformer);
                if (provider.Capabilities.ApiFormat == ApiFormat.Ollama)
                {
                    Console.WriteLine($"[UPSTREAM] → ollama model=\"{upstreamModel}\" id={id}");
                    bool handled = await TryHandleOllamaCloudChatCompletion(
                        ctx, provider, candidateBody, effectiveModel, upstreamModel, requestCt, clientCt, id, sw);
                    if (handled)
                    {
                        Console.WriteLine($"[RESP] 200 upstream={provider.Name} id={id}");
                        return;
                    }

                    Console.WriteLine($"[UPSTREAM] ← (ollama failover) id={id}");
                    continue;
                }

                string upstreamUrl = $"{provider.BaseUrl.TrimEnd('/')}/{provider.Capabilities.ChatPath.TrimStart('/')}";
                Console.WriteLine($"[UPSTREAM] → {upstreamUrl} model=\"{upstreamModel}\" id={id}");

                using StringContent content = new(candidateBody, Encoding.UTF8, "application/json");
                HttpRequestMessage request = new(HttpMethod.Post, provider.Capabilities.ChatPath) { Content = content };
                ApplyOpenRouterSessionHeader(request, provider, sessionRoot);
                HttpResponseMessage response = await provider.Client.SendAsync(request, requestCt);

                string responseBody = await response.Content.ReadAsStringAsync(clientCt);
                if (response.IsSuccessStatusCode)
                {
                    responseBody = NormalizeCompletionFinishReasons(responseBody);
                    reasoningCache.CacheReasoningFromResponse(responseBody);
                    ctx.Response.StatusCode = (int)response.StatusCode;
                    ctx.Response.ContentType = "application/json";
                    await ctx.Response.WriteAsync(responseBody, clientCt);
                    response.Dispose();
                    int outputTokens = TryGetUsageTokens(responseBody);
                    Console.WriteLine($"[RESP] 200 upstream={provider.Name} bytes={responseBody.Length} outtok={outputTokens} {sw.ElapsedMilliseconds}ms id={id}");
                    return;
                }

                Console.WriteLine($"[UPSTREAM] ← {(int)response.StatusCode} id={id} body={responseBody.Length}B");
                lastResponse?.Dispose();
                lastResponse = response;
                lastBody = responseBody;
            }

            ctx.Response.StatusCode = lastResponse is not null ? (int)lastResponse.StatusCode : StatusCodes.Status502BadGateway;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(lastBody ?? "{\"error\":\"no provider candidate available\"}", clientCt);
            Console.WriteLine($"[RESP] failover-exhausted status={ctx.Response.StatusCode} {sw.ElapsedMilliseconds}ms id={id}");
        }
        finally
        {
            lastResponse?.Dispose();
        }
    }

    private static async Task HandleStreamingCompletionAsync(
        HttpContext ctx,
        (ProviderInfo Provider, string UpstreamModel) candidate,
        string requestBody,
        string effectiveModel,
        RequestTransformer requestTransformer,
        ChatStreamingService chatStreaming,
        string sessionRoot,
        CancellationToken requestCt,
        CancellationToken clientCt,
        string id,
        Stopwatch sw)
    {
        ProviderInfo provider = candidate.Provider;
        string upstreamModel = candidate.UpstreamModel;
        string body = PrepareUpstreamRequestBody(requestBody, upstreamModel, effectiveModel, provider, requestTransformer);

        if (provider.Capabilities.ApiFormat == ApiFormat.Ollama)
        {
            Console.WriteLine($"[UPSTREAM] → ollama model=\"{upstreamModel}\" id={id}");
            await HandleOllamaCloudChatCompletion(ctx, provider, body, effectiveModel, upstreamModel, true, requestCt, clientCt, id, sw);
            return;
        }

        PrepareSseResponse(ctx);
        using StringContent content = new(body, Encoding.UTF8, "application/json");
        using HttpRequestMessage request = new(HttpMethod.Post, provider.Capabilities.ChatPath)
        {
            Content = content,
            Version = HttpVersion.Version11,
            VersionPolicy = HttpVersionPolicy.RequestVersionExact
        };
        ApplyOpenRouterSessionHeader(request, provider, sessionRoot);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        string upstreamUrl = $"{provider.BaseUrl.TrimEnd('/')}/{provider.Capabilities.ChatPath.TrimStart('/')}";
        Console.WriteLine($"[UPSTREAM] → {upstreamUrl} model=\"{upstreamModel}\" id={id}");

        using HttpResponseMessage response = await provider.Client.SendAsync(
            request, HttpCompletionOption.ResponseHeadersRead, requestCt);
        if (!response.IsSuccessStatusCode)
        {
            string errorBody = await response.Content.ReadAsStringAsync(clientCt);
            Console.WriteLine($"[UPSTREAM] ERROR ← {(int)response.StatusCode} id={id} body={errorBody.Length}B {sw.ElapsedMilliseconds}ms");
            ctx.Response.StatusCode = (int)response.StatusCode;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(errorBody, clientCt);
            return;
        }

        Console.WriteLine($"[UPSTREAM] ← {(int)response.StatusCode} id={id}");
        Console.WriteLine($"[STREAM] start id={id}");
        int outputTokens = await chatStreaming.StreamAndCache(
            response, ctx.Response, clientCt, effectiveModel, provider.Name, id);
        Console.WriteLine($"[STREAM] end id={id} outtok={outputTokens} {sw.ElapsedMilliseconds}ms");
    }

    private static string PrepareUpstreamRequestBody(
        string requestBody,
        string upstreamModel,
        string effectiveModel,
        ProviderInfo provider,
        RequestTransformer requestTransformer)
    {
        string body = requestTransformer.ReplaceModelInRequestBody(requestBody, upstreamModel);
        return requestTransformer.ApplyExecutionDefaults(body, effectiveModel, provider.Capabilities);
    }

    /// <summary>
    /// 从入站请求头提取会话级标识，用于把同一 Agent 会话的多轮 /v1/chat/completions
    /// 请求归并到同一个 sid。优先顺序：
    ///   1. X-Copilot-Chat-Conversation-Id  (VS Copilot 会话 id)
    ///   2. X-Github-Session-Id              (GitHub 会话 id)
    ///   3. traceparent                      (OpenTelemetry 标准 trace id)
    /// 都没有时返回空串，由调用方回退到随机串作为会话根。
    /// </summary>
    private static string ExtractSessionId(HttpContext ctx)
    {
        IHeaderDictionary headers = ctx.Request.Headers;

        if (headers.TryGetValue("X-Copilot-Chat-Conversation-Id", out var conv) && !string.IsNullOrWhiteSpace(conv))
            return conv.ToString().Trim();
        if (headers.TryGetValue("X-Github-Session-Id", out var gh) && !string.IsNullOrWhiteSpace(gh))
            return gh.ToString().Trim();
        if (headers.TryGetValue("traceparent", out var tp) && !string.IsNullOrWhiteSpace(tp))
        {
            // traceparent 格式: version-traceid-spanid-traceflags；取 traceid 段作为会话标识。
            string[] parts = tp.ToString().Split('-');
            if (parts.Length >= 2 && !string.IsNullOrWhiteSpace(parts[1]))
                return parts[1];
        }

        return string.Empty;
    }

    /// <summary>
    /// 为流式 SSE 响应设置标准响应头（状态码、Content-Type、禁用缓冲）。
    /// 在 OpenAI 流式分支与 Ollama 流式回退中复用，避免重复设置。
    /// </summary>
    private static void PrepareSseResponse(HttpContext ctx)
    {
        ctx.Response.StatusCode = 200;
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.Headers["X-Accel-Buffering"] = "no";
    }

    /// <summary>
    /// 将一个对象序列化为 OpenAI SSE chunk 并写入响应（data: {json}\n\n）。
    /// </summary>
    private static Task WriteSseChunkAsync(HttpContext ctx, object chunk, CancellationToken ct)
        => ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk, JsonDefaults.SnakeCase)}\n\n", ct);

    /// <summary>
    /// 从非流式 OpenAI 响应体的 <c>usage.completion_tokens</c> 提取输出 token 数。
    /// 缺失或解析失败时返回 0（不阻塞主流程）。
    /// </summary>
    private static int TryGetUsageTokens(string body)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            JsonElement root = doc.RootElement;
            if (root.TryGetProperty("usage", out JsonElement usage) && usage.ValueKind == JsonValueKind.Object
                && usage.TryGetProperty("completion_tokens", out JsonElement ct) && ct.ValueKind == JsonValueKind.Number)
            {
                return ct.GetInt32();
            }
        }
        catch
        {
            // 解析失败忽略
        }

        return 0;
    }

    private static string NormalizeCompletionFinishReasons(string json)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("choices", out JsonElement choices) || choices.ValueKind != JsonValueKind.Array)
                return json;

            using MemoryStream buffer = new();
            using (Utf8JsonWriter writer = new(buffer))
            {
                writer.WriteStartObject();
                foreach (JsonProperty property in root.EnumerateObject())
                {
                    if (!property.NameEquals("choices"))
                    {
                        property.WriteTo(writer);
                        continue;
                    }

                    writer.WritePropertyName("choices");
                    writer.WriteStartArray();
                    foreach (JsonElement choice in choices.EnumerateArray())
                    {
                        if (choice.ValueKind != JsonValueKind.Object ||
                            !choice.TryGetProperty("finish_reason", out JsonElement finishReason) ||
                            finishReason.ValueKind != JsonValueKind.String ||
                            IsOpenAiFinishReason(finishReason.GetString()))
                        {
                            choice.WriteTo(writer);
                            continue;
                        }

                        string? value = finishReason.GetString();
                        Console.WriteLine($"[RESP] normalized unsupported finish_reason=\"{value}\" to \"stop\"");
                        writer.WriteStartObject();
                        foreach (JsonProperty choiceProperty in choice.EnumerateObject())
                        {
                            if (choiceProperty.NameEquals("finish_reason"))
                                writer.WriteString("finish_reason", "stop");
                            else
                                choiceProperty.WriteTo(writer);
                        }
                        writer.WriteEndObject();
                    }
                    writer.WriteEndArray();
                }
                writer.WriteEndObject();
            }

            return Encoding.UTF8.GetString(buffer.ToArray());
        }
        catch (JsonException)
        {
            return json;
        }
    }

    private static bool IsOpenAiFinishReason(string? value)
        => value is "stop" or "length" or "content_filter" or "tool_calls" or "function_call";

    /// <summary>
    /// Resolves an explicit "provider/model" hint from the request id to a
    /// <see cref="ProviderInfo"/>. Returns null when the hint is absent, ambiguous,
    /// or points at a provider the registry does not know about.
    /// </summary>
    private static ProviderInfo? ExtractProviderHint(string? requestedModel, ProviderRegistry providerRegistry)
    {
        if (string.IsNullOrWhiteSpace(requestedModel))
            return null;

        int slash = requestedModel.IndexOf('/');
        if (slash <= 0 || slash >= requestedModel.Length - 1)
            return null;

        string providerHint = requestedModel[..slash];
        foreach (ProviderInfo prov in providerRegistry.Providers)
        {
            if (string.Equals(prov.Name, providerHint, StringComparison.OrdinalIgnoreCase))
                return prov;
        }

        return null;
    }

    /// <summary>
    /// 当上游 provider 是 OpenRouter 且客户端提供了会话 id 时，设置 x-session-id
    /// 头以启用 OpenRouter 的 sticky routing（粘性路由）。使用会话根（不含时分秒），
    /// 保证同一会话的多轮请求路由到同一 provider。
    /// </summary>
    private static void ApplyOpenRouterSessionHeader(HttpRequestMessage request, ProviderInfo provider, string? sessionId)
    {
        if (provider.Name.Equals("openrouter", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(sessionId))
        {
            request.Headers.TryAddWithoutValidation("x-session-id", sessionId);
        }
    }

    /// <summary>
    /// Attempts an Ollama Cloud chat completion as part of failover.
    /// Returns true if the response was written to the client; false if the candidate failed and the caller should try the next one.
    /// </summary>
    private static async Task<bool> TryHandleOllamaCloudChatCompletion(
        HttpContext ctx,
        ProviderInfo provider,
        string openAiRequestBody,
        string effectiveModel,
        string upstreamModel,
        CancellationToken requestCt,
        CancellationToken clientCt,
        string id,
        Stopwatch sw)
    {
        string ollamaRequestBody = BuildOllamaChatRequest(openAiRequestBody, upstreamModel, isStream: false);
        (HttpStatusCode statusCode, string responseBody) = await SendOllamaChatAsync(
            provider, ollamaRequestBody, requestCt, clientCt);
        if (!IsSuccessStatusCode(statusCode))
        {
            Console.WriteLine($"[UPSTREAM] ← {(int)statusCode} id={id} body={responseBody.Length}B");
            return false;
        }

        Console.WriteLine($"[UPSTREAM] ← {(int)statusCode} id={id}");
        string openAiResponseBody = ConvertOllamaChatToOpenAiCompletion(responseBody, effectiveModel);
        ctx.Response.StatusCode = 200;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(openAiResponseBody, clientCt);
        return true;
    }

    private static async Task HandleOllamaCloudChatCompletion(
        HttpContext ctx,
        ProviderInfo provider,
        string openAiRequestBody,
        string effectiveModel,
        string upstreamModel,
        bool isStream,
        CancellationToken requestCt,
        CancellationToken clientCt,
        string id,
        Stopwatch sw)
    {
        string ollamaRequestBody = BuildOllamaChatRequest(openAiRequestBody, upstreamModel, isStream: false);
        (HttpStatusCode statusCode, string responseBody) = await SendOllamaChatAsync(
            provider, ollamaRequestBody, requestCt, clientCt);
        if (!IsSuccessStatusCode(statusCode))
        {
            Console.WriteLine($"[UPSTREAM] ERROR ← {(int)statusCode} id={id} body={responseBody.Length}B {sw.ElapsedMilliseconds}ms");
            ctx.Response.StatusCode = (int)statusCode;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(responseBody, clientCt);
            return;
        }

        Console.WriteLine($"[UPSTREAM] ← {(int)statusCode} id={id}");
        string openAiResponseBody = ConvertOllamaChatToOpenAiCompletion(responseBody, effectiveModel);

        using JsonDocument completionDoc = JsonDocument.Parse(openAiResponseBody);
        JsonElement msg = completionDoc.RootElement.GetProperty("choices")[0].GetProperty("message");
        string contentText = msg.TryGetProperty("content", out JsonElement ce) && ce.ValueKind == JsonValueKind.String
            ? ce.GetString() ?? string.Empty
            : string.Empty;

        if (!isStream)
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(openAiResponseBody, clientCt);
            int outTok = TryGetUsageTokens(openAiResponseBody);
            Console.WriteLine($"[RESP] 200 upstream={provider.Name} outtok={outTok} {sw.ElapsedMilliseconds}ms id={id}");
            return;
        }

        // Streaming: Ollama Cloud non-streaming -> SSE chunks
        Console.WriteLine($"[STREAM] start id={id}");
        object firstChunk = new
        {
            id = $"chatcmpl-{Guid.NewGuid():N}",
            @object = "chat.completion.chunk",
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model = effectiveModel,
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { role = "assistant", content = contentText },
                    finish_reason = (string?)null
                }
            }
        };

        object finishChunk = new
        {
            id = $"chatcmpl-{Guid.NewGuid():N}",
            @object = "chat.completion.chunk",
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model = effectiveModel,
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { },
                    finish_reason = "stop"
                }
            }
        };

        PrepareSseResponse(ctx);

        await WriteSseChunkAsync(ctx, firstChunk, clientCt);
        await WriteSseChunkAsync(ctx, finishChunk, clientCt);
        await ctx.Response.WriteAsync("data: [DONE]\n\n", clientCt);
        Console.WriteLine($"[STREAM] end id={id} {sw.ElapsedMilliseconds}ms");
    }

    private static async Task<(HttpStatusCode StatusCode, string ResponseBody)> SendOllamaChatAsync(
        ProviderInfo provider,
        string requestBody,
        CancellationToken requestCt,
        CancellationToken clientCt)
    {
        using StringContent content = new(requestBody, Encoding.UTF8, "application/json");
        using HttpRequestMessage request = new(HttpMethod.Post, provider.Capabilities.ChatPath) { Content = content };
        using HttpResponseMessage response = await provider.Client.SendAsync(request, requestCt);
        string responseBody = await response.Content.ReadAsStringAsync(clientCt);
        return (response.StatusCode, responseBody);
    }

    private static bool IsSuccessStatusCode(HttpStatusCode statusCode)
        => (int)statusCode is >= StatusCodes.Status200OK and < StatusCodes.Status300MultipleChoices;

    private static string BuildOllamaChatRequest(string openAiRequestBody, string model, bool isStream)
    {
        using JsonDocument openAiDoc = JsonDocument.Parse(openAiRequestBody);
        JsonElement root = openAiDoc.RootElement;

        using MemoryStream ms = new();
        using Utf8JsonWriter writer = new(ms);

        writer.WriteStartObject();
        writer.WriteString("model", model);
        writer.WriteBoolean("stream", isStream);

        // ── Messages ──
        // Convert OpenAI multi-part content (text + image_url parts) into Ollama format:
        //   - text parts → "content" string
        //   - image_url parts → "images" array of base64 data URLs
        if (root.TryGetProperty("messages", out JsonElement messages))
        {
            writer.WritePropertyName("messages");
            writer.WriteStartArray();

            foreach (JsonElement msg in messages.EnumerateArray())
            {
                writer.WriteStartObject();

                bool hasMultiPartContent = false;
                List<string> imageUrls = [];

                // Determine if content is multi-part array with images
                if (msg.TryGetProperty("content", out JsonElement content) && content.ValueKind == JsonValueKind.Array)
                {
                    hasMultiPartContent = true;
                    StringBuilder textContent = new();

                    foreach (JsonElement part in content.EnumerateArray())
                    {
                        if (part.TryGetProperty("type", out JsonElement type) && type.GetString() == "text")
                        {
                            if (part.TryGetProperty("text", out JsonElement text) && text.ValueKind == JsonValueKind.String)
                            {
                                if (textContent.Length > 0)
                                    textContent.Append('\n');
                                textContent.Append(text.GetString());
                            }
                        }
                        else if (type.GetString() == "image_url")
                        {
                            if (part.TryGetProperty("image_url", out JsonElement imgUrl) && imgUrl.ValueKind == JsonValueKind.Object)
                            {
                                if (imgUrl.TryGetProperty("url", out JsonElement url) && url.ValueKind == JsonValueKind.String)
                                {
                                    imageUrls.Add(url.GetString()!);
                                }
                            }
                        }
                    }

                    writer.WriteString("content", textContent.ToString());
                }

                // Copy remaining properties (role, tool_calls, etc.) but skip content if already written
                // Also sanitize any invalid 'role' values coming from clients: some clients may
                // set role="tool", which OpenAI-style upstreams reject unless it's a tool
                // response tied to a preceding tool_calls entry. Replace such roles with
                // "assistant" to avoid API errors.
                foreach (JsonProperty mp in msg.EnumerateObject())
                {
                    if (mp.NameEquals("content") && hasMultiPartContent)
                        continue; // already written

                    if (mp.NameEquals("role") && mp.Value.ValueKind == JsonValueKind.String)
                    {
                        string? roleVal = mp.Value.GetString();
                        if (string.Equals(roleVal, "tool", StringComparison.OrdinalIgnoreCase))
                        {
                            writer.WriteString("role", "assistant");
                            continue;
                        }
                    }

                    mp.WriteTo(writer);
                }

                // Write images array if any image_url parts were found
                if (imageUrls.Count > 0)
                {
                    writer.WritePropertyName("images");
                    writer.WriteStartArray();
                    foreach (string imgUrl in imageUrls)
                    {
                        writer.WriteStringValue(imgUrl);
                    }
                    writer.WriteEndArray();
                }

                writer.WriteEndObject();
            }

            writer.WriteEndArray();
        }

        if (root.TryGetProperty("tools", out JsonElement tools))
        {
            writer.WritePropertyName("tools");
            tools.WriteTo(writer);
        }

        bool hasTemperature = root.TryGetProperty("temperature", out JsonElement temp);
        bool hasTopP = root.TryGetProperty("top_p", out JsonElement topP);
        bool hasMaxTokens = root.TryGetProperty("max_tokens", out JsonElement maxTokens);

        if (hasTemperature || hasTopP || hasMaxTokens)
        {
            writer.WritePropertyName("options");
            writer.WriteStartObject();
            if (hasTemperature && temp.ValueKind == JsonValueKind.Number)
            {
                writer.WriteNumber("temperature", temp.GetDouble());
            }

            if (hasTopP && topP.ValueKind == JsonValueKind.Number)
            {
                writer.WriteNumber("top_p", topP.GetDouble());
            }

            if (hasMaxTokens && maxTokens.ValueKind == JsonValueKind.Number)
            {
                writer.WriteNumber("num_predict", maxTokens.GetInt32());
            }

            writer.WriteEndObject();
        }

        writer.WriteEndObject();
        writer.Flush();
        return Encoding.UTF8.GetString(ms.ToArray());
    }

    private static string ConvertOllamaChatToOpenAiCompletion(string ollamaResponseBody, string effectiveModel)
    {
        using JsonDocument ollamaDoc = JsonDocument.Parse(ollamaResponseBody);
        JsonElement root = ollamaDoc.RootElement;
        JsonElement message = root.TryGetProperty("message", out JsonElement msg) ? msg : default;

        string content = message.ValueKind == JsonValueKind.Object && message.TryGetProperty("content", out JsonElement contentElement)
            ? contentElement.GetString() ?? string.Empty
            : string.Empty;

        // Fallback to `thinking` when content is empty (reasoning models put text in `thinking`).
        if (string.IsNullOrWhiteSpace(content) && message.ValueKind == JsonValueKind.Object && message.TryGetProperty("thinking", out JsonElement thinkingElement))
        {
            content = thinkingElement.GetString() ?? string.Empty;
        }

        object completion = new
        {
            id = $"chatcmpl-{Guid.NewGuid():N}",
            @object = "chat.completion",
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model = effectiveModel,
            choices = new[]
            {
                new
                {
                    index = 0,
                    message = new
                    {
                        role = "assistant",
                        content,
                        tool_calls = message.ValueKind == JsonValueKind.Object && message.TryGetProperty("tool_calls", out JsonElement tcs)
                            ? tcs
                            : (JsonElement?)null
                    },
                    finish_reason = root.TryGetProperty("done_reason", out JsonElement dr) && dr.ValueKind == JsonValueKind.String
                        ? dr.GetString()
                        : "stop"
                }
            }
        };

        return JsonSerializer.Serialize(completion, JsonDefaults.SnakeCase);
    }
}