// Async tests for debugging demonstrations
// These tests use tokio::test and demonstrate the breakpoint issue
// where breakpoints in async code don't hit properly

/// Async function that performs a computation
/// This function is designed to have clear breakpoint target lines
pub async fn async_computation(value: i32) -> i32 {
    let step1 = value + 10;  // Breakpoint target line (line 8)
    let step2 = step1 * 2;
    step2
}

/// Async function that performs multiple steps
pub async fn multi_step_async(x: i32, y: i32) -> i32 {
    let sum = x + y;          // Breakpoint target line (line 15)
    let product = x * y;      // Breakpoint target line (line 16)
    let result = sum + product;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_breakpoint() {
        let result = async_computation(5).await;
        assert_eq!(result, 30);
    }

    #[tokio::test]
    async fn test_multi_step_async() {
        let result = multi_step_async(3, 4).await;
        assert_eq!(result, 19); // (3 + 4) + (3 * 4) = 7 + 12 = 19
    }

    #[tokio::test]
    async fn test_async_with_multiple_awaits() {
        let result1 = async_computation(5).await;
        let result2 = async_computation(10).await;
        let combined = result1 + result2;
        assert_eq!(combined, 70); // 30 + 40 = 70
    }
}
