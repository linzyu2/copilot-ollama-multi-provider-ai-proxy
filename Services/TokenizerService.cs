using System.Reflection;
using System.Text.Json;
using Microsoft.DeepDev;

/// <summary>
/// 提供基于微软官方 BPE 分词器（Microsoft.DeepDev.TokenizerLib）的精确 Token 计算能力。
/// 由于该库原生只支持 OpenAI 系列编码器（cl100k_base / p50k / r50k / gpt2），
/// 对于其它模型家族（Llama、DeepSeek、Qwen 等）采用保守的字符启发式作为回退，
/// 以保证上下文裁剪始终偏向安全（高估 Token 数）。
/// </summary>
internal sealed class TokenizerService
{
    // cl100k_base 是覆盖面最广的现代编码器，被绝大多数 OpenAI 兼容模型使用，
    // 也是本项目默认采用的精确分词器。其 BPE rank 文件已作为内嵌资源打包，
    // 避免运行时从网络下载导致的不确定性。
    private static readonly IReadOnlyDictionary<string, int> Cl100kSpecialTokens =
        new Dictionary<string, int>
        {
            { "<|endoftext|>", 100257 },
            { "<|fim_prefix|>", 100258 },
            { "<|fim_middle|>", 100259 },
            { "<|fim_suffix|>", 100260 },
            { "<|endofprompt|>", 100276 },
        };

    private readonly ITokenizer _cl100k;

    public TokenizerService()
    {
        _cl100k = CreateCl100kTokenizer();
    }

    /// <summary>
    /// 计算给定文本的 Token 数量。优先使用精确的 cl100k_base 分词器；
    /// 当分词器不可用时回退到保守的字符启发式（字符数 / 2）。
    /// </summary>
    internal int CountTokens(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return 0;
        }

        try
        {
            return _cl100k.Encode(text, true).Count;
        }
        catch
        {
            // 分词器异常时回退到保守启发式，避免破坏请求处理。
            return text.Length / 2;
        }
    }

    /// <summary>
    /// 估算一条消息的 Token 数量，包含角色标记与结构化字段的近似开销。
    /// 对 content、reasoning_content、tool_calls 等字段分别精确计数。
    /// </summary>
    internal int CountMessageTokens(JsonElement message)
    {
        // 角色与消息框架的近似固定开销（约 4 个 token）。
        int total = 4;

        if (message.TryGetProperty("role", out JsonElement role) && role.ValueKind == JsonValueKind.String)
        {
            total += CountTokens(role.GetString()!);
        }

        if (message.TryGetProperty("content", out JsonElement content))
        {
            total += CountContentTokens(content);
        }

        if (message.TryGetProperty("reasoning_content", out JsonElement rc) && rc.ValueKind == JsonValueKind.String)
        {
            total += CountTokens(rc.GetString()!);
        }

        if (message.TryGetProperty("tool_calls", out JsonElement toolCalls) && toolCalls.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement tc in toolCalls.EnumerateArray())
            {
                if (tc.TryGetProperty("function", out JsonElement fn) && fn.ValueKind == JsonValueKind.Object)
                {
                    if (fn.TryGetProperty("name", out JsonElement fnName) && fnName.ValueKind == JsonValueKind.String)
                    {
                        total += CountTokens(fnName.GetString()!);
                    }
                    if (fn.TryGetProperty("arguments", out JsonElement args))
                    {
                        total += args.ValueKind == JsonValueKind.String
                            ? CountTokens(args.GetString()!)
                            : CountTokens(args.GetRawText());
                    }
                }
            }
        }

        return total;
    }

    private int CountContentTokens(JsonElement content)
    {
        return content.ValueKind switch
        {
            JsonValueKind.String => CountTokens(content.GetString()!),
            JsonValueKind.Array => CountContentPartsTokens(content),
            _ => 0
        };
    }

    private int CountContentPartsTokens(JsonElement parts)
    {
        int total = 0;
        foreach (JsonElement part in parts.EnumerateArray())
        {
            if (part.TryGetProperty("text", out JsonElement text) && text.ValueKind == JsonValueKind.String)
            {
                total += CountTokens(text.GetString()!);
            }
            else if (part.TryGetProperty("image_url", out JsonElement _))
            {
                // 图像按固定开销估算（约 85 token / 图），仅作保守近似。
                total += 85;
            }
        }
        return total;
    }

    private static ITokenizer CreateCl100kTokenizer()
    {
        Assembly assembly = typeof(TokenizerService).Assembly;
        // 通过文件名后缀定位内嵌资源，避免对根命名空间（随项目名变化）的硬编码依赖。
        string? resourceName = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith("cl100k_base.tiktoken", StringComparison.OrdinalIgnoreCase));
        using Stream? stream = resourceName is null ? null : assembly.GetManifestResourceStream(resourceName);
        if (stream is null)
        {
            // 资源缺失时回退到运行时下载（需要网络）。
            return TokenizerBuilder.CreateByEncoderNameAsync("cl100k_base", Cl100kSpecialTokens).GetAwaiter().GetResult();
        }

        const string pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        return TokenizerBuilder.CreateTokenizer(stream, Cl100kSpecialTokens, pattern);
    }
}
