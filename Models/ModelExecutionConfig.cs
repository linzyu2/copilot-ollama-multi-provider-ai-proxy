
/// <summary>
/// 描述从
/// <c>config/model-selection/*.json</c> 
/// 加载的提供程序/模型执行默认值和能力提示。
/// </summary>
/// <param name="ContextLength">模型所声明的最大输入上下文窗口。</param>
/// <param name="MaxOutputTokens">模型支持的最大输出令牌预算。</param>
/// <param name="SupportsTools">上游模型是否支持工具或函数调用。</param>
/// <param name="SupportsVision">上游模型是否接受图像或多模态输入。</param>
/// <param name="Family">用于路由和启发式判断的逻辑模型家族标签。</param>
/// <param name="Temperature">当客户端省略时注入的默认温度，或在启用覆盖时强制使用的温度。</param>
/// <param name="TopP">适用于目标提供程序/模型时注入的默认核采样值。</param>
/// <param name="MaxTokensPreferred">代理作为安全默认值使用时优先采用的 <c>max_tokens</c> 请求值。</param>
/// <param name="ReasoningEffort">支持该功能的提供程序/模型的可选推理努力提示。</param>
/// <param name="TimeoutSeconds">可选参数，用于按模型设置上游请求超时时间（以秒为单位）。</param>
/// <param name="MaxConcurrency">可选参数，用于限制该模型在代理内同时转发到上游的最大请求数。</param>
/// <param name="OverrideClientParams">当为 true 时，配置的请求参数将替换客户端提供的值，而不仅仅是填充缺失的参数。</param>
/// <param name="SupportsReasoning">该模型是否已知具备显式推理能力。</param>
public record struct ModelExecutionConfig(
    int? ContextLength = null,
    int? MaxOutputTokens = null,
    bool? SupportsTools = null,
    bool? SupportsVision = null,
    string? Family = null,
    double? Temperature = null,
    double? TopP = null,
    int? MaxTokensPreferred = null,
    string? ReasoningEffort = null,
    int? TimeoutSeconds = null,
    int? MaxConcurrency = null,
    bool OverrideClientParams = false,
    bool? SupportsReasoning = null
);
