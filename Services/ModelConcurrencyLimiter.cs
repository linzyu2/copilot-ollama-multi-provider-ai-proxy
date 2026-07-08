using System.Collections.Concurrent;

internal sealed class ModelConcurrencyLimiter
{
    private readonly ConcurrentDictionary<string, Gate> _gates = new(StringComparer.OrdinalIgnoreCase);

    internal ValueTask<Lease> AcquireAsync(string model, int? maxConcurrency, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(model) || !maxConcurrency.HasValue || maxConcurrency.Value <= 0)
        {
            return ValueTask.FromResult(Lease.Noop);
        }

        Gate gate = _gates.GetOrAdd(model, static (_, limit) => new Gate(limit), maxConcurrency.Value);
        return AcquireCoreAsync(gate, cancellationToken);
    }

    private static async ValueTask<Lease> AcquireCoreAsync(Gate gate, CancellationToken cancellationToken)
    {
        await gate.Semaphore.WaitAsync(cancellationToken);
        return new Lease(gate.Semaphore);
    }

    private sealed class Gate
    {
        internal Gate(int maxConcurrency)
        {
            MaxConcurrency = maxConcurrency;
            Semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
        }

        internal int MaxConcurrency { get; }
        internal SemaphoreSlim Semaphore { get; }
    }

    internal sealed class Lease : IAsyncDisposable
    {
        private readonly SemaphoreSlim? _semaphore;
        private int _released;

        internal static Lease Noop { get; } = new(null);

        internal Lease(SemaphoreSlim? semaphore)
        {
            _semaphore = semaphore;
        }

        public ValueTask DisposeAsync()
        {
            if (_semaphore is not null && Interlocked.Exchange(ref _released, 1) == 0)
            {
                _semaphore.Release();
            }

            return ValueTask.CompletedTask;
        }
    }
}