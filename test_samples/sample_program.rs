// Sample Rust program for debugging demonstrations
use std::env;

#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
}

impl Person {
    fn new(name: String, age: u32) -> Self {
        Person { name, age }
    }

    fn greet(&self) {
        println!("Hello, my name is {} and I'm {} years old", self.name, self.age);
    }

    fn have_birthday(&mut self) {
        self.age += 1;
        println!("{} is now {} years old!", self.name, self.age);
    }
}

fn factorial(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn process_numbers(numbers: Vec<i32>) -> i32 {
    let mut sum = 0;
    for num in numbers.iter() {
        sum += num;
    }
    sum
}

fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

fn trigger_panic() {
    panic!("This is a deliberate panic for testing!");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    println!("Rust debugging sample program");
    println!("Arguments: {:?}", args);
    
    // Create a person
    let mut person = Person::new("Alice".to_string(), 30);
    person.greet();
    person.have_birthday();
    
    // Calculate factorial
    let n = 5;
    let fact = factorial(n);
    println!("Factorial of {} is {}", n, fact);
    
    // Calculate fibonacci
    let fib_n = 10;
    let fib = fibonacci(fib_n);
    println!("Fibonacci of {} is {}", fib_n, fib);
    
    // Process a list of numbers
    let numbers = vec![1, 2, 3, 4, 5];
    let sum = process_numbers(numbers);
    println!("Sum of numbers is {}", sum);

    // Test data for debugging tools
    // Fixed-size array for print_array testing
    let fixed_array: [i32; 5] = [10, 20, 30, 40, 50];
    println!("Fixed array: {:?}", fixed_array);

    // Slice reference for print_array testing
    let slice: &[i32] = &fixed_array[1..4];
    println!("Slice (elements 1-3): {:?}", slice);

    // Mutable variable for watchpoint testing
    let mut watch_counter = 0;
    println!("Initial watch_counter: {}", watch_counter);

    // Loop to modify watch_counter (for watchpoint testing)
    for i in 1..=5 {
        watch_counter += i;
        println!("Loop iteration {}: watch_counter = {}", i, watch_counter);
    }

    // Mutable variable for set_variable testing
    let mut test_value = 42;
    println!("Initial test_value: {}", test_value);

    // Multiple distinct lines for continue_to_line testing
    test_value += 10;
    println!("After adding 10: test_value = {}", test_value);

    test_value *= 2;
    println!("After multiplying by 2: test_value = {}", test_value);

    test_value -= 20;
    println!("After subtracting 20: test_value = {}", test_value);

    // Use all variables to prevent optimization
    let array_sum: i32 = fixed_array.iter().sum();
    let slice_sum: i32 = slice.iter().sum();
    let combined_result = array_sum + slice_sum + watch_counter + test_value;
    println!("Combined result: {} + {} + {} + {} = {}",
             array_sum, slice_sum, watch_counter, test_value, combined_result);

    // Test division
    match divide(10, 2) {
        Ok(result) => println!("10 / 2 = {}", result),
        Err(e) => println!("Error: {}", e),
    }
    
    // Test division by zero
    match divide(10, 0) {
        Ok(result) => println!("10 / 0 = {}", result),
        Err(e) => println!("Error: {}", e),
    }
    
    // Trigger panic if requested
    if args.len() > 1 && args[1] == "panic" {
        trigger_panic();
    }
    
    println!("Program completed successfully!");
}