import React, { useState, useEffect } from "react"; // import react and useState. useEffect for running fuction whenever a state changes. 
import './App.css';
// import components
import Form from './components/Form'
import TodoList from './components/TodoList'

function App() {
  // === state data ===
  // value and function to set text
  const [inputText, setInputText] = useState("");
  const [todos, setTodos] = useState([]);
  const [status, setStatus] = useState('all');
  const [filteredTodos, setFilteredTodos] = useState([]);
  // === end state data ===

  // Run once when app is started
  useEffect(() => {
    getLocalTodos();
  }, []);

  useEffect(()=> {
    filterHandler();
    saveLocalTodos();
  }, [todos, status]) // everytime todos, status changes, run this function. 
  const filterHandler = () => {
    switch(status) {
      case 'completed': 
        setFilteredTodos(todos.filter(todo => (
          todo.completed === true
        )))
        break;
      case 'uncompleted': 
        setFilteredTodos(todos.filter(todo => (
          todo.completed === false
        )))
        break;
      default:
        setFilteredTodos(todos);
        break;
    }
  }

  // save to local 
  const saveLocalTodos = () => {
    localStorage.setItem("todos", JSON.stringify(todos));
  }

  // get todos from memory
  const getLocalTodos = () => {
    if(localStorage.getItem('todos') === null) {
      localStorage.setItem("todos", JSON.stringify([]));
    } else {
      const todoLocal = JSON.parse(localStorage.getItem("todos", JSON.stringify(todos)));
      setTodos(todoLocal)
    }
  }

  return (
    <div className="App">
      <header>
        <h1>My Todo List</h1>
        <p>{inputText}</p>
      </header>
      {/*
        Pass function to form using props. 
      */}
      <Form 
        inputText={inputText} 
        setInputText={setInputText} 
        todos={todos} 
        setTodos={setTodos} 
        setStatus={setStatus}
        />
      <TodoList 
        todos={todos} 
        setTodos={setTodos} 
        filteredTodos={filteredTodos}/> 
    </div>
  );
}

export default App;
