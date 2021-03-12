import React from 'react';

/** Create form component */
const Form = ({inputText, setInputText, todos, setTodos, setStatus}) => {

    const inputTextHandler = (e) => {
        console.log(e.target.value);
        setInputText(e.target.value);
    };

    const submitTodoHandler = (e) => {
        // prevent default behaviour (i.e. refresh.)
        e.preventDefault();
        setTodos([
            ...todos,       // existing todos
            {text: inputText, completed: false, id: Math.random() * 1000 }  // add new todo item.
        ]);
        // set input text state to empty string.
        setInputText("");
    };

    const statusHandler = (e) => {
        const state = e.target.value
        setStatus(state);
    }

    return (
        <form>
            <input value={inputText} onChange={inputTextHandler} type="text" className="todo-input"/>
            <button onClick={submitTodoHandler} className="todo-button" type="submit">
                <i class="fas fa-plus-square"></i>
            </button>    
            <div className="select">
                <select onChange={statusHandler} name="todos" className="filter-todo">
                    <option value="all">All</option>
                    <option value="completed">Completed</option>
                    <option value="uncompleted">Uncompleted</option>
                </select>
            </div>
        </form>
    )
};

// export form to hook it to app
export default Form;


