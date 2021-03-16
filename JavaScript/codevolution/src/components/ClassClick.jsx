import React, { Component } from 'react'

class ClassClick extends Component {

    handleClick() {
        console.log('I have been clicked')
    }

    render() {
        return (
            <button onClick={this.handleClick}>Click me</button>
        )
    }
}

export default ClassClick
