import React, { useState, useEffect } from 'react'

function HookCounterOne() {
    
    const [count, setCount] = useState(0)
    const [name, setName] = useState('')

    // executed after every render of dom.
    useEffect(() => {
        console.log('Use effect update title')
        document.title = `Clicked ${count} times`

        return () => {
            console.log('Unmounting code.')
        }
    }, [count]) // specify that effect is only called with count is updated. 
    

    /* 
        useEffect can mount, run, and demount code.
    */ 

    return (
        <div>
            <input type="text" value={name} onChange={e => setName(e.target.value)}/>
            <button onClick={() => setCount(count+1)}>Count: {count}</button>
        </div>
    )
}

export default HookCounterOne
