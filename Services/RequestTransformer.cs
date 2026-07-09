using System.Text;
using System.Text.Json;

internal sealed class RequestTransformer
{
    private readonly ModelCatalogService _modelCatalogService;
    private readonly ReasoningCacheService _reasoningCacheService;
    private readonly TokenizerService _tokenizerService;

    public RequestTransformer(ModelCatalogService modelCatalogService, ReasoningCacheService reasoningCacheService, TokenizerService tokenizerService)
    {
        _modelCatalogService = modelCatalogService;
        _reasoningCacheService = reasoningCacheService;
        _tokenizerService = tokenizerService;
    }

    internal string ApplyExecutionDefaults(string rawBody, string model, ProviderCapabilities capabilities = default)
    {
        ModelExecutionConfig exec = _modelCatalogService.GetExecutionConfigForModel(model);

        // 防御性滑动窗口裁剪：当历史上下文 + 当前请求的 Token 总量超过模型上下文窗口时，
        // 从最旧的非系统消息开始删除，始终保留系统消息，并为输出预留空间。
        rawBody = PruneToContextWindow(rawBody, exec);

        bool hasAnyDefault = exec.Temperature.HasValue
            || exec.TopP.HasValue
            || exec.MaxTokensPreferred.HasValue
            || !string.IsNullOrWhiteSpace(exec.ReasoningEffort);
        if (!hasAnyDefault)
        {
            return rawBody;
        }

        // Parameter support is driven by the provider's declared capabilities.
        // For unknown providers (default capabilities), all feature flags are false
        // (no reasoning_effort, no top_k) — a safe, conservative default.
        bool supportsReasoningEffort = capabilities.SupportsReasoningEffort;

        // OpenRouter (and similar aggregators) accept reasoning via a nested
        // `reasoning: { effort: "..." }` object rather than the OpenAI-style
        // top-level `reasoning_effort` field. When this is true we emit the
        // nested object and suppress the top-level field to avoid upstream 400s.
        bool supportsReasoningObject = capabilities.SupportsReasoningObject;

        // DeepSeek & OpenAI o-series: sending both temperature and top_p simultaneously
        // causes undefined behaviour per official docs. When reasoning_effort is active
        // (i.e. the model is a native reasoner) only emit temperature, not top_p.
        bool isNativeReasoner = capabilities.SupportsReasoningEffort && !string.IsNullOrWhiteSpace(exec.ReasoningEffort);

        // Providers that DO support top_k: NVIDIA, Groq, OpenRouter (passthrough).
        // Everyone else (including unknown) gets top_k stripped.
        bool supportsTopK = capabilities.SupportsTopK;

        // OverrideClientParams=true means the configured value is non-negotiable for this
        // model (e.g. Kimi K2.x requires temperature=1.0). In that mode we overwrite the
        // client-supplied field instead of only injecting defaults.
        bool force = exec.OverrideClientParams;

        // Context-aware budget: estimate input tokens, then cap max_tokens so
        // that input + output never exceeds the model's context window.
        // This prevents 400 errors from upstream when long conversations
        // reduce the available token budget.
        int? maxTokensByContext = null;
        if (exec.ContextLength.HasValue)
        {
            int estimatedInput = EstimateInputTokens(rawBody);
            maxTokensByContext = Math.Max(1, exec.ContextLength.Value - estimatedInput);
        }

        // Shared helper: clamp a candidate max_tokens value to the lowest of
        // MaxOutputTokens and the remaining context budget.
        static int ClampToBudget(int value, int? maxOutputTokens, int? maxByContext)
        {
            if (maxOutputTokens.HasValue && value > maxOutputTokens.Value)
                value = maxOutputTokens.Value;
            if (maxByContext.HasValue && value > maxByContext.Value)
                value = maxByContext.Value;
            return value;
        }

        try
        {
            using JsonDocument original = JsonDocument.Parse(rawBody);
            JsonElement root = original.RootElement;
            using MemoryStream ms = new();
            using Utf8JsonWriter writer = new(ms);

            writer.WriteStartObject();

            bool hasTemperature = false;
            bool hasTopP = false;
            bool hasMaxTokens = false;
            bool hasReasoningEffort = false;

            foreach (JsonProperty prop in root.EnumerateObject())
            {
                if (prop.NameEquals("temperature"))
                {
                    if (force && exec.Temperature.HasValue)
                    {
                        writer.WriteNumber("temperature", exec.Temperature.Value);
                    }
                    else
                    {
                        prop.WriteTo(writer);
                    }
                    hasTemperature = true;
                }
                else if (prop.NameEquals("top_p"))
                {
                    if (force && exec.TopP.HasValue)
                    {
                        writer.WriteNumber("top_p", exec.TopP.Value);
                    }
                    else
                    {
                        prop.WriteTo(writer);
                    }
                    hasTopP = true;
                }
                else if (prop.NameEquals("max_tokens"))
                {
                    if (force && exec.MaxTokensPreferred.HasValue)
                    {
                        int clamped = ClampToBudget(exec.MaxTokensPreferred.Value, exec.MaxOutputTokens, maxTokensByContext);
                        writer.WriteNumber("max_tokens", clamped);
                    }
                    else
                    {
                        // Dynamic clamp: cap client-supplied max_tokens to model's output budget and context window.
                        int clientVal = prop.Value.GetInt32();
                        int clamped = ClampToBudget(clientVal, exec.MaxOutputTokens, maxTokensByContext);
                        if (clamped != clientVal)
                        {
                            writer.WriteNumber("max_tokens", clamped);
                        }
                        else
                        {
                            prop.WriteTo(writer);
                        }
                    }
                    hasMaxTokens = true;
                }
                else if (prop.NameEquals("reasoning_effort"))
                {
                    if (supportsReasoningObject)
                    {
                        // OpenRouter uses a nested `reasoning` object instead of the
                        // top-level field; drop the client-supplied value here and
                        // re-emit it via the nested object at the end.
                        hasReasoningEffort = true;
                        continue;
                    }

                    if (force && !string.IsNullOrWhiteSpace(exec.ReasoningEffort))
                    {
                        writer.WriteString("reasoning_effort", exec.ReasoningEffort);
                    }
                    else
                    {
                        prop.WriteTo(writer);
                    }
                    hasReasoningEffort = true;
                }
                else if (prop.NameEquals("top_k") && !supportsTopK)
                {
                    // Skip top_k for providers that don't support it
                    continue;
                }
                else
                {
                    prop.WriteTo(writer);
                }
            }

            if (!hasTemperature && exec.Temperature.HasValue)
                writer.WriteNumber("temperature", exec.Temperature.Value);
            // Skip top_p injection for native reasoners to avoid temperature+top_p conflict.
            if (!hasTopP && exec.TopP.HasValue && !isNativeReasoner)
                writer.WriteNumber("top_p", exec.TopP.Value);
            if (!hasMaxTokens && exec.MaxTokensPreferred.HasValue)
            {
                int clamped = ClampToBudget(exec.MaxTokensPreferred.Value, exec.MaxOutputTokens, maxTokensByContext);
                writer.WriteNumber("max_tokens", clamped);
            }
            // Only inject reasoning_effort for providers/models that natively support it.
            if (!hasReasoningEffort && !string.IsNullOrWhiteSpace(exec.ReasoningEffort) && supportsReasoningEffort)
                writer.WriteString("reasoning_effort", exec.ReasoningEffort);

            // OpenRouter-style nested reasoning object: `reasoning: { effort: "..." }`.
            // Emitted when the provider supports it and a reasoning effort is configured,
            // regardless of whether the client already sent a top-level reasoning_effort
            // (which we stripped above to avoid the unsupported field).
            if (!string.IsNullOrWhiteSpace(exec.ReasoningEffort) && supportsReasoningObject)
            {
                writer.WriteStartObject("reasoning");
                writer.WriteString("effort", exec.ReasoningEffort);
                writer.WriteEndObject();
            }

            writer.WriteEndObject();
            writer.Flush();

            return Encoding.UTF8.GetString(ms.ToArray());
        }
        catch
        {
            return rawBody;
        }
    }

    /// <summary>
    /// Estimates the number of input tokens from the raw request body using the
    /// precise tokenizer. Falls back to a conservative heuristic on failure.
    /// </summary>
    private int EstimateInputTokens(string rawBody)
    {
        return _tokenizerService.CountTokens(rawBody);
    }

    /// <summary>
    /// 防御性滑动窗口裁剪：当请求的总输入 Token 数（含 messages、tools、system 等）
    /// 超过模型上下文窗口时，从最旧的非系统消息开始删除，始终保留系统消息，
    /// 并为输出预留 <paramref name="exec"/> 中声明的 max_tokens 预算。
    /// 若裁剪后仍然超限，则继续删除非系统消息，直至满足约束或仅剩系统消息。
    /// </summary>
    private string PruneToContextWindow(string rawBody, ModelExecutionConfig exec)
    {
        if (!exec.ContextLength.HasValue)
        {
            return rawBody;
        }

        int contextLength = exec.ContextLength.Value;
        // 为输出预留空间：优先使用配置的输出预算，否则预留 25% 上下文窗口。
        int reservedOutput = exec.MaxOutputTokens ?? Math.Max(1, contextLength / 4);
        int inputBudget = Math.Max(1, contextLength - reservedOutput);

        try
        {
            using JsonDocument doc = JsonDocument.Parse(rawBody);
            JsonElement root = doc.RootElement;
            if (!root.TryGetProperty("messages", out JsonElement messages) || messages.ValueKind != JsonValueKind.Array)
            {
                return rawBody;
            }

            // 计算当前总 Token 数（messages + 其它顶层字段）。
            int totalTokens = CountNonMessageTokens(root, messages);
            List<JsonElement> msgList = [];
            foreach (JsonElement m in messages.EnumerateArray())
            {
                msgList.Add(m);
            }

            // 分离系统消息（始终保留）与非系统消息（可裁剪）。
            List<JsonElement> systemMessages = [];
            List<JsonElement> prunable = [];
            foreach (JsonElement m in msgList)
            {
                string? role = m.TryGetProperty("role", out JsonElement r) ? r.GetString() : null;
                if (string.Equals(role, "system", StringComparison.OrdinalIgnoreCase))
                {
                    systemMessages.Add(m);
                }
                else
                {
                    prunable.Add(m);
                }
            }

            foreach (JsonElement m in systemMessages)
            {
                totalTokens += _tokenizerService.CountMessageTokens(m);
            }
            foreach (JsonElement m in prunable)
            {
                totalTokens += _tokenizerService.CountMessageTokens(m);
            }

            if (totalTokens <= inputBudget)
            {
                return rawBody;
            }

            // 从最旧的非系统消息开始删除（prunable 列表按原始顺序排列）。
            int i = 0;
            while (totalTokens > inputBudget && i < prunable.Count)
            {
                totalTokens -= _tokenizerService.CountMessageTokens(prunable[i]);
                prunable.RemoveAt(i);
            }

            // 重建消息数组：系统消息在前，保留的非系统消息在后。
            List<JsonElement> finalMessages = [.. systemMessages, .. prunable];

            using MemoryStream ms = new();
            using Utf8JsonWriter writer = new(ms);
            writer.WriteStartObject();
            foreach (JsonProperty prop in root.EnumerateObject())
            {
                if (prop.NameEquals("messages"))
                {
                    continue;
                }
                prop.WriteTo(writer);
            }
            writer.WritePropertyName("messages");
            writer.WriteStartArray();
            foreach (JsonElement m in finalMessages)
            {
                m.WriteTo(writer);
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.Flush();

            return Encoding.UTF8.GetString(ms.ToArray());
        }
        catch
        {
            return rawBody;
        }
    }

    /// <summary>
    /// 计算请求中除 messages 数组以外的顶层字段（如 tools、system 字符串等）的 Token 数。
    /// </summary>
    private int CountNonMessageTokens(JsonElement root, JsonElement messages)
    {
        int total = 0;
        foreach (JsonProperty prop in root.EnumerateObject())
        {
            if (prop.NameEquals("messages"))
            {
                continue;
            }
            total += _tokenizerService.CountTokens(prop.Value.GetRawText());
        }
        return total;
    }

    internal string ReplaceModelInRequestBody(string rawBody, string upstreamModel)
    {
        try
        {
            using JsonDocument original = JsonDocument.Parse(rawBody);
            JsonElement root = original.RootElement;
            using MemoryStream ms = new();
            using Utf8JsonWriter writer = new(ms);

            writer.WriteStartObject();
            bool hasModel = false;

            foreach (JsonProperty prop in root.EnumerateObject())
            {
                if (prop.NameEquals("model"))
                {
                    writer.WriteString("model", upstreamModel);
                    hasModel = true;
                    continue;
                }

                prop.WriteTo(writer);
            }

            if (!hasModel)
            {
                writer.WriteString("model", upstreamModel);
            }

            writer.WriteEndObject();
            writer.Flush();

            return Encoding.UTF8.GetString(ms.ToArray());
        }
        catch
        {
            return rawBody;
        }
    }

    internal string? ModifyRequest(JsonDocument doc)
    {
        JsonElement root = doc.RootElement;
        if (!root.TryGetProperty("messages", out JsonElement msgs))
        {
            return null;
        }

        int idx = 0;
        bool modified = false;
        using MemoryStream ms = new();
        using Utf8JsonWriter w = new(ms);

        w.WriteStartObject();
        foreach (JsonProperty prop in root.EnumerateObject())
        {
            if (!prop.NameEquals("messages"))
            {
                prop.WriteTo(w);
                continue;
            }

            w.WritePropertyName("messages");
            w.WriteStartArray();
            foreach (JsonElement msg in msgs.EnumerateArray())
            {
                string? role = msg.TryGetProperty("role", out JsonElement r) ? r.GetString() : null;
                if (role == "assistant")
                {
                    bool hasTc = msg.TryGetProperty("tool_calls", out JsonElement tcArr) && tcArr.GetArrayLength() > 0;
                    bool hasFunctionCall = msg.TryGetProperty("function_call", out JsonElement fc)
                        && fc.ValueKind == JsonValueKind.Object;
                    string? key = null;

                    if (hasTc)
                    {
                        List<string> ids = [];
                        foreach (JsonElement tc in tcArr.EnumerateArray())
                            if (tc.TryGetProperty("id", out JsonElement idE) && idE.ValueKind == JsonValueKind.String)
                                ids.Add(idE.GetString()!);
                        if (ids.Count > 0) key = $"toolcall:{string.Join("|", ids)}";
                    }
                    else
                    {
                        key = $"assistant:{idx++}";
                    }

                    if (key != null && _reasoningCacheService.TryGet(key, out string? rc))
                    {
                        bool needsInject = !msg.TryGetProperty("reasoning_content", out JsonElement exRc)
                            || exRc.ValueKind != JsonValueKind.String
                            || string.IsNullOrEmpty(exRc.GetString());

                        if (needsInject)
                        {
                            w.WriteStartObject();
                            foreach (JsonProperty mp in msg.EnumerateObject())
                                mp.WriteTo(w);
                            w.WriteString("reasoning_content", rc);
                            w.WriteEndObject();
                            modified = true;
                            continue;
                        }
                    }

                    // Some providers (e.g. Moonshot/Kimi) reject assistant messages whose
                    // content is empty when there are no tool/function calls associated.
                    // Drop those invalid placeholders before forwarding upstream, but only
                    // when there is no cached reasoning to preserve for thinking models.
                    if (!hasTc && !hasFunctionCall && IsAssistantContentEmpty(msg))
                    {
                        modified = true;
                        continue;
                    }
                }

                msg.WriteTo(w);
            }

            w.WriteEndArray();
        }

        w.WriteEndObject();
        w.Flush();

        return modified ? Encoding.UTF8.GetString(ms.ToArray()) : null;
    }

    private static bool IsAssistantContentEmpty(JsonElement msg)
    {
        if (!msg.TryGetProperty("content", out JsonElement content))
            return true;

        return content.ValueKind switch
        {
            JsonValueKind.Null => true,
            JsonValueKind.Undefined => true,
            JsonValueKind.String => string.IsNullOrWhiteSpace(content.GetString()),
            JsonValueKind.Array => content.GetArrayLength() == 0,
            _ => false
        };
    }
}
