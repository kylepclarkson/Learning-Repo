import React, {useReducer} from 'react'

const initialState = {
    firstCounter: 0,
}
// state: current value of counter
// action: one of the three button actions.
const reducer = (state, action) => {

    switch(action.type) {
        case 'increment':
            return {firstCounter: state + 1};
        case 'decrement':
            return {firstCounter: state - 1};
        case 'reset':
            return initialState;
        default:
            return state;
    }
}

function CounterTwo() {

    // count, current state. 
    const [count, dispatch] = useReducer(reducer, initialState) 

    return (
        <div>
            <div>Count: {count.firstCounter}</div>
            <button onClick={() => dispatch({type: 'increment'})}>Increment</button>
            <button onClick={() => dispatch({type: 'decrement'})}>Decrement</button>
            <button onClick={() => dispatch({type: 'reset'})}>Reset</button>
        </div>
    )
}

export default CounterTwo
