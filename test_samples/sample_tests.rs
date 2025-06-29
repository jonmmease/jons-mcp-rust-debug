// Sample Rust tests for debugging demonstrations

#[derive(Debug, PartialEq)]
struct Calculator {
    memory: f64,
}

impl Calculator {
    fn new() -> Self {
        Calculator { memory: 0.0 }
    }

    fn add(&mut self, a: f64, b: f64) -> f64 {
        let result = a + b;
        self.memory = result;
        result
    }

    fn subtract(&mut self, a: f64, b: f64) -> f64 {
        let result = a - b;
        self.memory = result;
        result
    }

    fn multiply(&mut self, a: f64, b: f64) -> f64 {
        let result = a * b;
        self.memory = result;
        result
    }

    fn divide(&mut self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 {
            Err("Division by zero".to_string())
        } else {
            let result = a / b;
            self.memory = result;
            Ok(result)
        }
    }

    fn get_memory(&self) -> f64 {
        self.memory
    }

    fn clear_memory(&mut self) {
        self.memory = 0.0;
    }
}

fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    for i in 2..=(n as f64).sqrt() as u32 {
        if n % i == 0 {
            return false;
        }
    }
    true
}

fn reverse_string(s: &str) -> String {
    s.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_add() {
        let mut calc = Calculator::new();
        assert_eq!(calc.add(2.0, 3.0), 5.0);
        assert_eq!(calc.get_memory(), 5.0);
    }

    #[test]
    fn test_calculator_subtract() {
        let mut calc = Calculator::new();
        assert_eq!(calc.subtract(10.0, 3.0), 7.0);
        assert_eq!(calc.get_memory(), 7.0);
    }

    #[test]
    fn test_calculator_multiply() {
        let mut calc = Calculator::new();
        assert_eq!(calc.multiply(4.0, 5.0), 20.0);
        assert_eq!(calc.get_memory(), 20.0);
    }

    #[test]
    fn test_calculator_divide() {
        let mut calc = Calculator::new();
        assert_eq!(calc.divide(10.0, 2.0), Ok(5.0));
        assert_eq!(calc.get_memory(), 5.0);
    }

    #[test]
    fn test_calculator_divide_by_zero() {
        let mut calc = Calculator::new();
        assert_eq!(calc.divide(10.0, 0.0), Err("Division by zero".to_string()));
    }

    #[test]
    fn test_calculator_memory() {
        let mut calc = Calculator::new();
        calc.add(5.0, 3.0);
        assert_eq!(calc.get_memory(), 8.0);
        calc.clear_memory();
        assert_eq!(calc.get_memory(), 0.0);
    }

    #[test]
    fn test_is_prime() {
        assert_eq!(is_prime(0), false);
        assert_eq!(is_prime(1), false);
        assert_eq!(is_prime(2), true);
        assert_eq!(is_prime(3), true);
        assert_eq!(is_prime(4), false);
        assert_eq!(is_prime(17), true);
        assert_eq!(is_prime(100), false);
    }

    #[test]
    fn test_reverse_string() {
        assert_eq!(reverse_string("hello"), "olleh");
        assert_eq!(reverse_string("rust"), "tsur");
        assert_eq!(reverse_string(""), "");
        assert_eq!(reverse_string("a"), "a");
    }

    #[test]
    fn test_complex_scenario() {
        let mut calc = Calculator::new();
        
        // Perform multiple operations
        let result1 = calc.add(10.0, 5.0);
        let result2 = calc.multiply(result1, 2.0);
        let result3 = calc.subtract(result2, 10.0);
        
        assert_eq!(result1, 15.0);
        assert_eq!(result2, 30.0);
        assert_eq!(result3, 20.0);
        assert_eq!(calc.get_memory(), 20.0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_deliberate_panic() {
        // This test is expected to panic
        assert!(false, "assertion failed");
    }
}