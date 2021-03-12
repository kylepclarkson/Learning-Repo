import React from 'react';
// import component
import Todo from "./Todo";

const TodoList = ({todos, setTodos, filteredTodos}) => {

    return (
        <div className="todo-container">
            <ul className="todo-list">
                {/* Render each todo component */}
                {filteredTodos.map(todo => (
                    <Todo 
                    setTodos={setTodos} 
                    todo = {todo}
                    todos = {todos}
                    key={todo.id} 
                    text={todo.text} />
                ))}
            </ul>
        </div>
    )
}

export default TodoList;