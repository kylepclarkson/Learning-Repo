import React from 'react';
// import component
import Todo from "./Todo";

const TodoList = ({todos}) => {

    return (
        <div className="todo-container">
            <ul className="todo-list">
                {/* Render each todo component */}
                {todos.map(todo => (
                    <Todo key={todo.id} text={todo.text} />
                ))}
            </ul>
        </div>
    )
}

export default TodoList;