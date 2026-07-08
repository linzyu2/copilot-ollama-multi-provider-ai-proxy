internal sealed class ProviderRegistry
{
    private readonly List<ProviderInfo> _providers = [];
    private Dictionary<string, ProviderInfo> _modelToProvider = new(StringComparer.OrdinalIgnoreCase);
    private Dictionary<string, string> _modelToUpstream = new(StringComparer.OrdinalIgnoreCase);
    private Dictionary<string, List<ProviderInfo>> _upstreamToProviders = new(StringComparer.OrdinalIgnoreCase);

    public ProviderRegistry(ProviderHttpClientFactory httpClientFactory)
    {
        DefaultModel = Environment.GetEnvironmentVariable("DEEPSEEK_MODEL") ?? "deepseek-v4-flash";
        DiscoverProviders(httpClientFactory);

        // ── Startup diagnostic: print all registered providers ──
        if (_providers.Count > 0)
        {
            Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  Registered Providers                                           ║");
            Console.WriteLine("╠══════════════════════════════════════════════════════════════════╣");
            foreach (ProviderInfo p in _providers)
            {
                string maskedKey = p.ApiKey.Length > 8
                    ? p.ApiKey[..8] + "…"
                    : p.ApiKey.Length > 0 ? "****" : "(no key)";
                Console.WriteLine($"║  {p.Name,-12} │ {p.BaseUrl,-46} ║");
                Console.WriteLine($"║  {' ',12} │ API Key: {maskedKey,-36} ║");
            }
            Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
        }
        else
        {
            Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  ⚠️  No se encontraron API keys configuradas.                    ║");
            Console.WriteLine("║                                                                  ║");
            Console.WriteLine("║  Edita el archivo .env en la raíz del proyecto y descomenta     ║");
            Console.WriteLine("║  al menos un proveedor, por ejemplo:                            ║");
            Console.WriteLine("║                                                                  ║");
            Console.WriteLine("║    PROVIDER_DEEPSEEK_API_KEY=sk-tu-key-aqui                      ║");
            Console.WriteLine("║                                                                  ║");
            Console.WriteLine("║  Luego reinicia la aplicación.                                   ║");
            Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
        }
    }

    internal string DefaultModel { get; }

    internal IReadOnlyList<ProviderInfo> Providers => _providers;

    internal IReadOnlyDictionary<string, ProviderInfo> ModelToProvider => _modelToProvider;

    internal IReadOnlyDictionary<string, string> ModelToUpstream => _modelToUpstream;

    internal void UpdateModelMappings(
        Dictionary<string, ProviderInfo> modelToProvider,
        Dictionary<string, string> modelToUpstream,
        Dictionary<string, List<ProviderInfo>>? upstreamToProviders = null)
    {
        _modelToProvider = modelToProvider;
        _modelToUpstream = modelToUpstream;

        if (upstreamToProviders is not null)
        {
            // Caller (ModelCatalogService) already ordered providers by configured priority.
            _upstreamToProviders = upstreamToProviders;
            return;
        }

        // Fallback: build from modelToProvider, preserving discovery order as a tie-break.
        Dictionary<string, List<ProviderInfo>> upstream = new(StringComparer.OrdinalIgnoreCase);
        foreach ((string modelId, ProviderInfo prov) in modelToProvider)
        {
            string up = modelToUpstream.TryGetValue(modelId, out string? u) ? u : modelId;
            if (!upstream.TryGetValue(up, out List<ProviderInfo>? list))
            {
                list = [];
                upstream[up] = list;
            }

            if (!list.Any(p => p.Name.Equals(prov.Name, StringComparison.OrdinalIgnoreCase)))
            {
                int order = _providers.FindIndex(p => p.Name.Equals(prov.Name, StringComparison.OrdinalIgnoreCase));
                int insertAt = list.FindIndex(p => _providers.FindIndex(q => q.Name.Equals(p.Name, StringComparison.OrdinalIgnoreCase)) > order);
                if (insertAt < 0)
                {
                    list.Add(prov);
                }
                else
                {
                    list.Insert(insertAt, prov);
                }
            }
        }

        _upstreamToProviders = upstream;
    }

    internal ProviderInfo ResolveProvider(string? requestedModel)
    {
        if (_providers.Count == 0)
            throw new InvalidOperationException("ProviderRegistry: no providers are registered (check PROVIDER_*_API_KEY env vars).");
        if (!string.IsNullOrWhiteSpace(requestedModel))
        {
            string cleanModel = NormalizeRequestedModel(requestedModel);
            if (_modelToProvider.TryGetValue(cleanModel, out ProviderInfo provider))
                return provider;
        }
        return _providers[0];
    }

    internal string ResolveModel(string? requestedModel)
    {
        if (_providers.Count == 0)
            return DefaultModel;

        if (!string.IsNullOrWhiteSpace(requestedModel))
        {
            string cleanModel = NormalizeRequestedModel(requestedModel);
            if (_modelToProvider.ContainsKey(cleanModel))
                return cleanModel;

            // Accept the OpenAI-style "provider/model" form (e.g. "groq/llama-3.3-70b-versatile"
                // or "nvidia/qwen3.5-397b-a17b"). Some providers (NVIDIA in particular) expose
                // upstream ids with a slash prefix ("qwen/qwen3.5-397b-a17b"), so first try the
                // full id verbatim, then fall back to stripping the provider prefix, and finally
                // try matching the requested bare against any upstream id owned by the hinted
                // provider that ends with that bare.
                int slash = cleanModel.IndexOf('/');
                if (slash > 0 && slash < cleanModel.Length - 1)
                {
                    if (_modelToProvider.ContainsKey(cleanModel))
                        return cleanModel;

                    string bare = cleanModel[(slash + 1)..];
                    if (_modelToProvider.ContainsKey(bare))
                        return bare;

                    // Last-resort match: look for any upstream id in the hinted provider
                    // whose suffix equals the requested bare (e.g. requested bare
                    // "qwen3.5-397b-a17b" matches upstream "qwen/qwen3.5-397b-a17b").
                    string? providerHint = cleanModel[..slash];
                    if (_modelToProvider.TryGetValue(bare, out _))
                    {
                        return bare; // unreachable, but keeps the flow explicit
                    }

                    foreach (KeyValuePair<string, ProviderInfo> kv in _modelToProvider)
                    {
                        if (!string.Equals(kv.Value.Name, providerHint, StringComparison.OrdinalIgnoreCase))
                            continue;
                        int lastSlash = kv.Key.LastIndexOf('/');
                        string suffix = lastSlash > 0 ? kv.Key[(lastSlash + 1)..] : kv.Key;
                        if (string.Equals(suffix, bare, StringComparison.OrdinalIgnoreCase))
                            return kv.Key;
                    }
                }

                // Handle the "model@provider" qualified form generated by NormalizeRequestedModel.
                // When "OPENROUTER - nemotron-3-ultra-550b-a55b:latest" is normalised to
                // "nemotron-3-ultra-550b-a55b@OPENROUTER" but the catalog only has
                // "nvidia/nemotron-3-ultra-550b-a55b@openrouter", search by provider + suffix.
                int atSign = cleanModel.IndexOf('@');
                if (atSign > 0 && atSign < cleanModel.Length - 1)
                {
                    string bareModel = cleanModel[..atSign];
                    string providerHint = cleanModel[(atSign + 1)..];

                    foreach (KeyValuePair<string, ProviderInfo> kv in _modelToProvider)
                    {
                        if (!string.Equals(kv.Value.Name, providerHint, StringComparison.OrdinalIgnoreCase))
                            continue;

                        // Strip "@provider" suffix from the key to get the upstream model id.
                        int keyAt = kv.Key.IndexOf('@');
                        string upstreamId = keyAt > 0 ? kv.Key[..keyAt] : kv.Key;

                        // Match when the upstream id ends with the bare model (e.g.
                        // "nvidia/nemotron-3-ultra-550b-a55b" ends with "nemotron-3-ultra-550b-a55b").
                        if (upstreamId.EndsWith(bareModel, StringComparison.OrdinalIgnoreCase))
                            return kv.Key;
                    }
                }
        }

        return DefaultModel;
    }

    /// <summary>Removes only the ephemeral ":latest" tag from an Ollama model name.</summary>
    /// <remarks>Does NOT strip other colons (e.g. ":free") which are semantically
    /// significant for providers like OpenRouter (pricing tier) or Ollama (quantization).</remarks>
    private static string StripTagSuffix(string model)
    {
        if (model.EndsWith(":latest", StringComparison.OrdinalIgnoreCase))
            return model[..^7];
        return model;
    }

    /// <summary>
    /// Normalizes model identifiers sent back by Ollama/BYOM clients.
    /// Accepts provider-qualified values (model@provider:latest), bare values
    /// (model:latest), and display labels such as "OPENROUTER - model:latest".
    /// When a display prefix matches a known provider name, the output is
    /// reconstructed as "model@provider" so routing preserves the intended
    /// provider instead of falling back to the default (highest-priority)
    /// provider for the bare model name.
    /// </summary>
    private string NormalizeRequestedModel(string model)
    {
        string clean = model.Trim();

        int displayPrefix = clean.IndexOf(" - ", StringComparison.Ordinal);
        if (displayPrefix > 0 && displayPrefix + 3 < clean.Length)
        {
            string rawProvider = clean[..displayPrefix].Trim();
            string rawModel = clean[(displayPrefix + 3)..].Trim();

            if (_providers.Any(p => p.Name.Equals(rawProvider, StringComparison.OrdinalIgnoreCase)))
            {
                string bareModel = StripTagSuffix(rawModel);
                string canonicalProvider = _providers.First(p => p.Name.Equals(rawProvider, StringComparison.OrdinalIgnoreCase)).Name;
                return $"{bareModel}@{canonicalProvider}";
            }

            clean = rawModel;
        }

        return StripTagSuffix(clean);
    }

    internal string ResolveUpstreamModel(string? requestedModel)
    {
        string resolved = ResolveModel(requestedModel);
        return _modelToUpstream.TryGetValue(resolved, out string? upstream) ? upstream : resolved;
    }

    /// <summary>
    /// Returns the ordered list of (provider, upstreamModel) candidates for a requested model.
    /// A provider-qualified id ("model@provider") resolves to a single candidate (no failover).
    /// A bare model id returns every provider that offers it, ordered by configured priority.
    /// </summary>
    internal IReadOnlyList<(ProviderInfo Provider, string UpstreamModel)> ResolveCandidates(string? requestedModel)
    {
        string resolved = ResolveModel(requestedModel);
        string upstream = _modelToUpstream.TryGetValue(resolved, out string? up) ? up : resolved;

        // Explicit "model@provider" alias -> single, exact candidate.
        bool isQualified = resolved.Contains('@');
        if (isQualified && _modelToProvider.TryGetValue(resolved, out ProviderInfo exact))
        {
            return [(exact, upstream)];
        }

        if (_upstreamToProviders.TryGetValue(upstream, out List<ProviderInfo>? providers) && providers.Count > 0)
        {
            return providers.Select(p => (p, upstream)).ToArray();
        }

        // Empty registry → return no candidates; ResolveProvider would throw.
        if (_providers.Count == 0)
            return [];
        return [(ResolveProvider(resolved), upstream)];
    }

    private void DiscoverProviders(ProviderHttpClientFactory httpClientFactory)
    {
        foreach (string providerName in ProviderCapabilitiesRegistry.KnownProviders)
        {
            ProviderCapabilities caps = ProviderCapabilitiesRegistry.Get(providerName);
            string prefix = caps.EnvPrefix;

            string? apiKey = Environment.GetEnvironmentVariable($"PROVIDER_{prefix}_API_KEY");

            // Backward compatibility: Ollama Cloud also accepts PROVIDER_OLLAMA_API_KEY.
            if (string.IsNullOrWhiteSpace(apiKey) && providerName == "ollama")
            {
                apiKey = Environment.GetEnvironmentVariable("PROVIDER_OLLAMA_API_KEY");
            }

            // Skip providers that require an API key when none is configured.
            // Ollama can still be discovered later as a local instance (no API key needed).
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                continue;
            }

            string? configuredBaseUrl = Environment.GetEnvironmentVariable($"PROVIDER_{prefix}_BASE_URL");
            string baseUrl = string.IsNullOrWhiteSpace(configuredBaseUrl)
                ? caps.DefaultBaseUrl
                : configuredBaseUrl;

            HttpClient provClient = httpClientFactory.CreateProviderClient(providerName, baseUrl, apiKey);
            _providers.Add(new ProviderInfo(providerName, apiKey, baseUrl, provClient, caps));
        }

        // ── Local Ollama (no API key required) ──────────────────────────
        // When PROVIDER_OLLAMA_BASE_URL points to localhost / 127.0.0.1 / ::1,
        // register a second "ollama" entry that works without an API key.
        string? ollamaBaseUrl = Environment.GetEnvironmentVariable("PROVIDER_OLLAMA_BASE_URL");
        if (!string.IsNullOrWhiteSpace(ollamaBaseUrl)
            && (ollamaBaseUrl.Contains("localhost", StringComparison.OrdinalIgnoreCase)
                || ollamaBaseUrl.Contains("127.0.0.1")
                || ollamaBaseUrl.Contains("::1"))
            && !_providers.Any(p => p.BaseUrl.Equals(ollamaBaseUrl, StringComparison.OrdinalIgnoreCase)))
        {
            ProviderCapabilities ollamaCaps = ProviderCapabilitiesRegistry.Get("ollama");
            HttpClient localClient = httpClientFactory.CreateProviderClient("ollama", ollamaBaseUrl, apiKey: "");
            _providers.Add(new ProviderInfo("ollama", "", ollamaBaseUrl, localClient, ollamaCaps));
        }

        // ── Legacy DEEPSEEK_API_KEY fallback ────────────────────────────
        string? legacyKey = Environment.GetEnvironmentVariable("DEEPSEEK_API_KEY");
        if (!string.IsNullOrWhiteSpace(legacyKey) && !_providers.Any(p => p.Name == "deepseek"))
        {
            ProviderCapabilities deepseekCaps = ProviderCapabilitiesRegistry.Get("deepseek");
            string legacyUrl = Environment.GetEnvironmentVariable("DEEPSEEK_BASE_URL") ?? deepseekCaps.DefaultBaseUrl;
            _providers.Add(new ProviderInfo("deepseek", legacyKey, legacyUrl,
                httpClientFactory.CreateProviderClient("deepseek", legacyUrl, legacyKey), deepseekCaps));
        }
    }
}
