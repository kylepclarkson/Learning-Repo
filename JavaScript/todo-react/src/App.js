import React, { useState } from "react"; // import react and useState 
import './App.css';
// import components
import Form from './components/Form'
import TodoList from './components/TodoList'

function App() {
  // value and function to set text
  const [inputText, setInputText] = useState("");
  
  return (
    <div className="App">
      <header>
        <h1>My Todo List</h1>
        <p>{inputText}</p>
      </header>
      {/*
        Pass function to form using props. 
      */}
      <Form setInputText={setInputText}/>
      <TodoList/> 
    </div>
  );
}

export default App;
