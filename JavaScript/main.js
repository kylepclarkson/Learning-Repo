
// varables: var, let, const
// var age = 30;
// var ted = "40";
// console.log("The age: " + age);
// console.log("Double age: " + 2*age);

// data types
// String, Numbers, Boolean, null, undefined 

// const name = 'John';
// const age = 30;
// const rating = 4.5;
// const flag = true;
// const x= null;
// const y= undefined;
// let z;

// console.log(typeof name);
// console.log(typeof age);
// console.log(typeof rating);
// console.log(typeof flag);
// console.log(typeof x);
// console.log(typeof y);
// console.log(typeof z);

// === Concat variables and text ===
// const name = 'John';
// const age = 30;
// // Template String
// console.log(`My name is ${name} and I am ${age}`);

// const s = "Hello world";
// console.log(s.length)
// console.log(s.toLowerCase())
// console.log(s.substring(0, 5))
// let x = "Ted, Alex, Meg, Thunder"
// console.log(x.split(",")) 

// === Arrays ===
// const numbers = new Array(1,2,3,4,5);
// console.log(numbers);

// const fruits = ['Apple', 'Fruits', 'Pears', 55, 55, 100]
// console.log(fruits);
// console.log(fruits[0]);
// console.log(fruits[3]);
// fruits[3] += 20;
// console.log(fruits[3]);
// console.log(fruits)

// === object literals (Key-value pairs) ===
const person = {
    firstName: 'John',
    lastName: 'Doe',
    age: 30,
    address: {
        street: 'main st',
        city: 'Brandon',
        state: 'MB'
    }
}
person.site = 'email.com'
console.log(person)
console.log(person.firstName)
console.log(person.address.state)

const todos = [
    {
        id: 1,
        text: "Sweep",
        isCompleted: false,
    },
    {
        id:2,
        text: "Run",
        isCompleted: false,
    },
    {
        id:3,
        text: "Meeting",
        isCompleted: true,
    },
]

// === for loops ===
for (let i=0; i<todos.length; i++) {
    console.log(todos[i]);
}

todos.forEach(function(todo) {
    console.log(todo);
})

const todoText = todos.map(function(todo) {
    return todo.text;
})
console.log(todoText)

const needTodos = todos.filter(function(todo) {
    return todo.isCompleted == false;
}).map(function(todo) {
    return todo.text;
})
console.log(needTodos);

