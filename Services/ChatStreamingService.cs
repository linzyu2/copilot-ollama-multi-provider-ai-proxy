using System.Text;
using System.Text.Json;

internal sealed class ChatStreamingService
{
    private readonly ReasoningCacheService _reasoningCacheService;

    public ChatStreamingService(ReasoningCacheService reasoningCacheService)
    {
        _reasoningCacheService = reasoningCacheService;
    }

    internal async Task<int> StreamAndCache(
        HttpResponseMessage upstream,
        HttpResponse downstream,
        CancellationToken ct,
        string model = "unknown",
        string provider = "unknown",
        string requestId = "unknown")
    {
        int completionTokens = 0;
        try
        {
            using Stream upstreamStream = await upstream.Content.ReadAsStreamAsync(ct);
            using StreamReader reader = new(upstreamStream);
            await using StreamWriter writer = new(downstream.Body, leaveOpen: true) { NewLine = "\n" };

            StringBuilder sb = new(4096);
            List<string>? tcIds = null;
            bool hasTc = false;
            string? assistantKey = null;
            bool receivedData = false;

            while (true)
            {
                string? line;
                try
                {
                    line = await reader.ReadLineAsync(ct);
                }
                catch (HttpIOException) when (receivedData)
                {
                    // Upstream closed the connection after we already sent data.
                    // This is a normal end of stream — break gracefully.
                    break;
                }

                if (line == null) break;
                receivedData = true;

                if (line.StartsWith("data:"))
                {
                    string json = line.Substring(5).TrimStart();
                    if (json.Length > 0 && json != "[DONE]")
                    {
                        try
                        {
                            using JsonDocument chunk = JsonDocument.Parse(json);
                            JsonElement cr = chunk.RootElement;
                            if (TryGetUpstreamError(cr, out string errorMessage, out string? errorCode))
                            {
                                Console.WriteLine(
                                    $"[STREAM] upstream error provider={provider} model=\"{model}\" id={requestId} " +
                                    $"code=\"{errorCode ?? "unknown"}\" message=\"{errorMessage}\"");
                                await WriteOpenAiErrorAsync(downstream, errorMessage, errorCode, ct);
                                break;
                            }

                            json = NormalizeFinishReason(json, cr);
                            if (cr.TryGetProperty("choices", out JsonElement choices) && choices.GetArrayLength() > 0)
                            {
                                JsonElement delta = choices[0].TryGetProperty("delta", out JsonElement d) ? d
                                    : choices[0].TryGetProperty("message", out JsonElement mm) ? mm : default;

                                if (delta.ValueKind != JsonValueKind.Undefined)
                                {
                                    if (cr.TryGetProperty("usage", out JsonElement usage) && usage.ValueKind == JsonValueKind.Object)
                                    {
                                        if (usage.TryGetProperty("completion_tokens", out JsonElement ctE) && ctE.ValueKind == JsonValueKind.Number)
                                            completionTokens = ctE.GetInt32();
                                    }

                                    if (delta.TryGetProperty("reasoning_content", out JsonElement rc) && rc.ValueKind == JsonValueKind.String)
                                    {
                                        string? rct = rc.GetString();
                                        if (!string.IsNullOrEmpty(rct)) sb.Append(rct);
                                    }

                                    if (delta.TryGetProperty("tool_calls", out JsonElement tcs) && tcs.ValueKind == JsonValueKind.Array)
                                    {
                                        hasTc = true;
                                        foreach (JsonElement tc in tcs.EnumerateArray())
                                        {
                                            if (tc.TryGetProperty("id", out JsonElement idE) && idE.ValueKind == JsonValueKind.String)
                                            {
                                                tcIds ??= [];
                                                string id = idE.GetString()!;
                                                if (!tcIds.Contains(id)) tcIds.Add(id);
                                            }
                                        }
                                    }

                                    if (choices[0].TryGetProperty("finish_reason", out JsonElement fr) && fr.ValueKind != JsonValueKind.Null)
                                    {
                                        string reasoning = sb.ToString();
                                        if (!string.IsNullOrEmpty(reasoning))
                                        {
                                            string key;
                                            if (hasTc && tcIds != null && tcIds.Count > 0)
                                                key = $"toolcall:{string.Join("|", tcIds)}";
                                            else
                                                key = assistantKey ??= _reasoningCacheService.NextAssistantKey();

                                            _reasoningCacheService.Set(key, reasoning);
                                        }
                                    }
                                }
                            }
                        }
                        catch
                        {
                            // parse errors are non-critical
                        }

                        await writer.WriteAsync("data: ");
                        await writer.WriteAsync(json);
                        await writer.WriteLineAsync();
                    }
                    else if (json == "[DONE]")
                    {
                        await writer.WriteLineAsync(line);
                    }
                }
                else
                {
                    await writer.WriteLineAsync(line);
                }

                await writer.FlushAsync(ct);
            }
        }
        catch (HttpIOException)
        {
            // Upstream connection dropped before any data was received.
            // The 200 header was already sent to the client, so we cannot switch
            // to a JSON error response. Simply end the stream — the client will
            // see an incomplete response and will retry automatically.
        }

        return completionTokens;
    }

    private static bool TryGetUpstreamError(JsonElement root, out string message, out string? code)
    {
        message = string.Empty;
        code = null;

        if (!root.TryGetProperty("error", out JsonElement error))
            return false;

        if (error.ValueKind == JsonValueKind.Object)
        {
            if (error.TryGetProperty("message", out JsonElement messageElement) &&
                messageElement.ValueKind == JsonValueKind.String)
            {
                message = messageElement.GetString() ?? string.Empty;
            }

            if (error.TryGetProperty("code", out JsonElement codeElement))
                code = codeElement.ToString();
        }
        else if (error.ValueKind == JsonValueKind.String)
        {
            message = error.GetString() ?? string.Empty;
        }

        if (string.IsNullOrWhiteSpace(message))
            message = "Upstream provider returned an error.";

        return true;
    }

    private static async Task WriteOpenAiErrorAsync(
        HttpResponse downstream,
        string message,
        string? code,
        CancellationToken ct)
    {
        Dictionary<string, object?> error = new()
        {
            ["message"] = message,
            ["type"] = "upstream_error",
            ["code"] = code
        };

        await downstream.WriteAsync(
            $"data: {JsonSerializer.Serialize(new { error }, JsonDefaults.SnakeCase)}\n\n", ct);
        await downstream.WriteAsync("data: [DONE]\n\n", ct);
    }

    private static string NormalizeFinishReason(string json, JsonElement root)
    {
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
                        finishReason.ValueKind != JsonValueKind.String)
                    {
                        choice.WriteTo(writer);
                        continue;
                    }

                    string? value = finishReason.GetString();
                    if (IsOpenAiFinishReason(value))
                    {
                        choice.WriteTo(writer);
                        continue;
                    }

                    Console.WriteLine($"[STREAM] normalized unsupported finish_reason=\"{value}\" to \"stop\"");
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

    private static bool IsOpenAiFinishReason(string? value)
        => value is "stop" or "length" or "content_filter" or "tool_calls" or "function_call";

    internal async Task StreamNdjsonPassthrough(HttpResponseMessage upstream, HttpResponse downstream, CancellationToken ct)
    {
        using Stream upstreamStream = await upstream.Content.ReadAsStreamAsync(ct);
        using StreamReader reader = new(upstreamStream);
        await using StreamWriter writer = new(downstream.Body, leaveOpen: true) { NewLine = "\n" };

        while (true)
        {
            string? line = await reader.ReadLineAsync(ct);
            if (line == null)
            {
                break;
            }

            await writer.WriteLineAsync(line);
            await writer.FlushAsync(ct);
        }
    }

    internal async Task StreamOllamaAndCache(HttpResponseMessage upstream, HttpResponse downstream, string model, CancellationToken ct)
    {
        using Stream upstreamStream = await upstream.Content.ReadAsStreamAsync(ct);
        using StreamReader reader = new(upstreamStream);
        await using StreamWriter writer = new(downstream.Body, leaveOpen: true) { NewLine = "\n" };

        StringBuilder reasoningSb = new(4096);
        List<string>? tcIds = null;
        bool hasTc = false;
        string finishReason = "stop";

        while (true)
        {
            string? line = await reader.ReadLineAsync(ct);
            if (line == null) break;
            if (!line.StartsWith("data:")) continue;

            string json = line.Substring(5).TrimStart();
            if (json.Length == 0 || json == "[DONE]") continue;

            string? contentDelta = null;
            JsonElement? toolCallsDelta = null;
            try
            {
                using JsonDocument chunk = JsonDocument.Parse(json);
                JsonElement cr = chunk.RootElement;
                if (cr.TryGetProperty("choices", out JsonElement choices) && choices.GetArrayLength() > 0)
                {
                    JsonElement choice0 = choices[0];
                    JsonElement delta = choice0.TryGetProperty("delta", out JsonElement d) ? d
                        : choice0.TryGetProperty("message", out JsonElement mm) ? mm : default;

                    if (delta.ValueKind != JsonValueKind.Undefined)
                    {
                        if (delta.TryGetProperty("content", out JsonElement ce) && ce.ValueKind == JsonValueKind.String)
                            contentDelta = ce.GetString();

                        if (delta.TryGetProperty("reasoning_content", out JsonElement rc) && rc.ValueKind == JsonValueKind.String)
                        {
                            string? rct = rc.GetString();
                            if (!string.IsNullOrEmpty(rct)) reasoningSb.Append(rct);
                        }

                        if (delta.TryGetProperty("tool_calls", out JsonElement tcs) && tcs.ValueKind == JsonValueKind.Array)
                        {
                            hasTc = true;
                            toolCallsDelta = tcs.Clone();
                            foreach (JsonElement tc in tcs.EnumerateArray())
                            {
                                if (tc.TryGetProperty("id", out JsonElement idE) && idE.ValueKind == JsonValueKind.String)
                                {
                                    tcIds ??= [];
                                    string id = idE.GetString()!;
                                    if (!tcIds.Contains(id)) tcIds.Add(id);
                                }
                            }
                        }
                    }

                    if (choice0.TryGetProperty("finish_reason", out JsonElement fr) && fr.ValueKind == JsonValueKind.String)
                    {
                        finishReason = fr.GetString() ?? "stop";
                        string reasoning = reasoningSb.ToString();
                        if (!string.IsNullOrEmpty(reasoning))
                        {
                            string key = hasTc && tcIds != null && tcIds.Count > 0
                                ? $"toolcall:{string.Join("|", tcIds)}"
                                : _reasoningCacheService.NextAssistantKey();
                            _reasoningCacheService.Set(key, reasoning);
                        }
                    }
                }
            }
            catch
            {
                continue;
            }

            if (contentDelta == null && toolCallsDelta == null) continue;

            Dictionary<string, object?> message = new()
            {
                ["role"] = "assistant",
                ["content"] = contentDelta ?? ""
            };
            if (toolCallsDelta != null)
                message["tool_calls"] = toolCallsDelta.Value;

            Dictionary<string, object?> ndjson = new()
            {
                ["model"] = model,
                ["created_at"] = DateTime.UtcNow.ToString("o"),
                ["message"] = message,
                ["done"] = false
            };

            await writer.WriteAsync(JsonSerializer.Serialize(ndjson, JsonDefaults.SnakeCase));
            await writer.WriteLineAsync();
            await writer.FlushAsync(ct);
        }

        Dictionary<string, object?> final = new()
        {
            ["model"] = model,
            ["created_at"] = DateTime.UtcNow.ToString("o"),
            ["message"] = new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "" },
            ["done_reason"] = finishReason,
            ["done"] = true
        };
        await writer.WriteAsync(JsonSerializer.Serialize(final, JsonDefaults.SnakeCase));
        await writer.WriteLineAsync();
        await writer.FlushAsync(ct);
    }
}
