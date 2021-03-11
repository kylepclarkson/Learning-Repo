import React from 'react';

/** Create form component */
const Form = ({setInputText}) => {

    const inputTextHandler = (e) => {
        console.log(e.target.value);
        setInputText(e.target.value);
    };

    return (
        <form>
            {/* Call handler when text changes */}
            <input onChange={inputTextHandler} type="text" className="todo-input"/>
            <button className="todo-button" type="submit">
                <i class="fas fa-plus-square"></i>
            </button>    
            <div className="select">
                <select name="todos" className="filter-todo">
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


