namespace ProxyTests;

[Collection("Proxy")]
public class ModelConcurrencyLimiterTests
{
    [Fact]
    public async Task AcquireAsync_WithMaxConcurrencyOne_BlocksUntilLeaseReleased()
    {
        ModelConcurrencyLimiter limiter = new();

        await using ModelConcurrencyLimiter.Lease firstLease = await limiter.AcquireAsync("nvidia/nemotron-3-super-120b-a12b:free", 1, CancellationToken.None);

        Task<ModelConcurrencyLimiter.Lease> secondAcquire = limiter.AcquireAsync("nvidia/nemotron-3-super-120b-a12b:free", 1, CancellationToken.None).AsTask();
        await Task.Delay(100);

        Assert.False(secondAcquire.IsCompleted);

        await firstLease.DisposeAsync();
        await using ModelConcurrencyLimiter.Lease secondLease = await secondAcquire.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.True(secondAcquire.IsCompletedSuccessfully);
    }

    [Fact]
    public async Task AcquireAsync_WithoutLimit_ReturnsImmediately()
    {
        ModelConcurrencyLimiter limiter = new();

        await using ModelConcurrencyLimiter.Lease lease = await limiter.AcquireAsync("openai/gpt-5.5", null, CancellationToken.None);

        Assert.NotNull(lease);
    }
}
