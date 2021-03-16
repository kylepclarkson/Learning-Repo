import React from 'react'

function FragmentDemo() {
    return (
        // use fragment to exclude the single div tag needed by React.
        // Can also use empty brackets (cannot pass key) <></>
        <React.Fragment>
            <h1>Fragement Demo</h1>
            <p>This is a demo fragment</p>
        </React.Fragment>
    )
}

export default FragmentDemo
