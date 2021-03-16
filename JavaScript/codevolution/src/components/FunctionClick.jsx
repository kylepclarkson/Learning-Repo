import React, { Component } from 'react'

function FunctionClick() {
    function clickHandler() {
        console.log("Button click.");
    }
    return (
        <div>
            <button onClick={clickHandler}>Click</button>
        </div>
    )
}

export default FunctionClick;
