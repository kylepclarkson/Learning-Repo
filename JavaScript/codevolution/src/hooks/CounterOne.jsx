import React, {useReducer} from 'react'


const initialState = 0
// state: current value of counter
// action: one of the three button actions.
const reducer = (state, action) => {

    switch(action) {
        case 'increment':
            return state + 1;
        case 'decrement':
            return state - 1;
        case 'reset':
            return initialState;
        default:
            return state;
    }
}

function CounterOne() {

    // count, current state. 
    const [count, dispatch] = useReducer(reducer, initialState) 

    return (
        <div>
            <div>Count: {count}</div>
            <button onClick={() => dispatch('increment')}>Increment</button>
            <button onClick={() => dispatch('decrement')}>Decrement</button>
            <button onClick={() => dispatch('reset')}>Reset</button>
        </div>
    )
}

export default CounterOne
