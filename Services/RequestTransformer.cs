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

        // OpenRouter prompt caching only applies to Claude models; decide once up front
        // so we can inject the breakpoint inside the single streaming pass below.
        bool needCache_control = NeedCache_control(model);

        // 防御性滑动窗口裁剪：当历史上下文 + 当前请求的 Token 总量超过模型上下文窗口时，
        // 从最旧的非系统消息开始删除，始终保留系统消息，并为输出预留空间。
        rawBody = PruneToContextWindow(rawBody, exec);

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

        // Context-aware budget: compute after pruning so the estimation always
        // uses the already-trimmed body. Capped at 1 to guarantee a non-zero
        // output budget, preventing corner-case clamping to 0 max_tokens.
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
            bool hasCacheControl = false;

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
                else if (prop.NameEquals("cache_control"))
                {
                    // Preserve any explicit top-level breakpoint the client already set.
                    prop.WriteTo(writer);
                    hasCacheControl = true;
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

            // OpenRouter prompt caching: for Claude models, inject a top-level
            // "cache_control" breakpoint (ttl "1h") to unlock the discounted prompt-cache
            // window. Done in this same pass so the body is scanned/written only once.
            // The "messages[].content" array is never touched, so explicit breakpoints the
            // client set there are preserved. Skip if the client already supplied one.
            if (needCache_control && !hasCacheControl)
            {
                writer.WriteStartObject("cache_control");
                writer.WriteString("type", "ephemeral");
                //writer.WriteString("ttl", "1h");
                writer.WriteEndObject();
            }

            writer.WriteEndObject();
            writer.Flush();

            return Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
        }
        catch (InvalidOperationException)
        {
            throw;
        }
        catch
        {
            return rawBody;
        }
    }

    /// <summary>
    /// Estimates the number of input tokens
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
    /// 受保护块（所有 system 消息、首个 user 消息）永不裁剪，以稳定 OpenRouter
    /// 的提示词缓存哈希；带 tool_calls 的 assistant 消息与其 tool 响应作为原子块
    /// 整体保留或删除，避免 tool_call 与 tool 响应失配。若裁剪后仍然超限，则继续
    /// 删除非系统消息，直至满足约束或仅剩受保护块。
    /// </summary>
    internal string PruneToContextWindow(string rawBody, ModelExecutionConfig exec)
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
            int totalTokens = _tokenizerService.CountNonMessageTokens(root, messages);
            List<JsonElement> msgList = [];
            foreach (JsonElement m in messages.EnumerateArray())
            {
                msgList.Add(m);
            }

            // 将消息分组为「裁剪块」：普通消息各自成块；带 tool_calls 的 assistant 消息
            // 与其紧随其后的 tool 响应合并为一个原子块，避免只删一半导致 tool_call 与
            // tool 响应失配（上游 API 会拒绝此类非法对话）。
            List<(List<JsonElement> Block, bool IsToolExchange)> blocks = [];
            for (int bi = 0; bi < msgList.Count; bi++)
            {
                JsonElement m = msgList[bi];
                string? role = m.TryGetProperty("role", out JsonElement r) ? r.GetString() : null;

                if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase)
                    && m.TryGetProperty("tool_calls", out JsonElement tc)
                    && tc.ValueKind == JsonValueKind.Array && tc.GetArrayLength() > 0)
                {
                    List<JsonElement> block = [m];
                    int j = bi + 1;
                    while (j < msgList.Count)
                    {
                        JsonElement nx = msgList[j];
                        string? nxRole = nx.TryGetProperty("role", out JsonElement nr) ? nr.GetString() : null;
                        if (string.Equals(nxRole, "tool", StringComparison.OrdinalIgnoreCase))
                        {
                            block.Add(nx);
                            j++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    blocks.Add((block, true));
                    bi = j - 1; // 跳过已合并的 tool 响应，避免重复成块
                }
                else
                {
                    blocks.Add(([m], false));
                }
            }

            // 钉死 OpenRouter 缓存哈希依赖的首个 system 与首个 user 消息所在块，永不裁剪。
            // 其余 system 块同样受保护（沿用「始终保留系统消息」的既有策略）。
            int pinnedUserBlock = -1;
            for (int k = 0; k < blocks.Count; k++)
            {
                foreach (JsonElement m in blocks[k].Block)
                {
                    string? r2 = m.TryGetProperty("role", out JsonElement rr) ? rr.GetString() : null;
                    if (pinnedUserBlock < 0 && string.Equals(r2, "user", StringComparison.OrdinalIgnoreCase))
                    {
                        pinnedUserBlock = k;
                    }
                }
                if (pinnedUserBlock >= 0)
                {
                    break;
                }
            }

            // 计算每块 token 数并累加至总量。
            List<int> blockTokens = new(blocks.Count);
            foreach (var (block, _) in blocks)
            {
                int t = 0;
                foreach (JsonElement m in block)
                {
                    t += _tokenizerService.CountMessageTokens(m);
                }
                blockTokens.Add(t);
                totalTokens += t;
            }

            if (totalTokens <= inputBudget)
            {
                return rawBody;
            }

            // 从最旧的块开始删除：跳过受保护块（所有 system 块 + 首个 user 块），
            // 且 tool 交换块整体删除，保证 tool_call 与 tool 响应配对完整。
            bool[] removed = new bool[blocks.Count];
            for (int k = 0; k < blocks.Count && totalTokens > inputBudget; k++)
            {
                bool isSystem = blocks[k].Block.Any(m =>
                    m.TryGetProperty("role", out JsonElement r3) && string.Equals(r3.GetString(), "system", StringComparison.OrdinalIgnoreCase));
                if (isSystem || k == pinnedUserBlock)
                {
                    continue;
                }
                totalTokens -= blockTokens[k];
                removed[k] = true;
            }

            // 防穿透熔断：即使删除所有非系统消息后仍然超限，说明 tools、response_format
            // 或系统消息本身已超出模型上下文窗口。比较对象使用完整的 contextLength
            // （而非 inputBudget，因为 inputBudget 在 max_output_tokens == context_length
            // 时可能低至 1），确保只在真正无药可救时熔断。
            if (totalTokens > contextLength)
            {
                throw new InvalidOperationException(
                    $"Request exceeds context window ({contextLength}) after removing all non-system messages. " +
                    "The system message and/or tool/response_format definitions alone exceed the available budget.");
            }

            // 重建消息数组：按原始顺序保留未被裁剪的块（块内消息顺序不变）。
            List<JsonElement> finalMessages = [];
            for (int k = 0; k < blocks.Count; k++)
            {
                if (removed[k])
                {
                    continue;
                }
                finalMessages.AddRange(blocks[k].Block);
            }

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

            return Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
        }
        catch (InvalidOperationException)
        {
            throw;
        }
        catch
        {
            return rawBody;
        }
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

            return Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
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

        return modified ? Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length) : null;
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

    private static bool NeedCache_control(string model) =>new[] { "anthropic/claude", "qwen/qwen" }.Any(prefix => model.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
}
