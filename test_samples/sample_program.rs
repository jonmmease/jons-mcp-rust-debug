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