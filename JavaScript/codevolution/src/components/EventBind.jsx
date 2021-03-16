import React, { Component } from 'react'

class EventBind extends Component {

    constructor(props) {
        super(props)

        this.state = {
            message: 'Hello'
        }
        // bind 'this' once in constructor
        // this.clickHandler = this.clickHandler.bind(this)
    }

    clickHandler() {
        // does not work; due to how this is used in javascript.
        this.setState({
            message: 'Goodbye'
        })
    }
    
    // class property approach. 
    clickHandler = () => {
        this.setState({
            message: 'Goodbye'
        })
    }

    render() {
        return (
            <div>
                <h3>{this.state.message}</h3>
                {/* <button onClick={this.clickHandler}>Click</button> */}
                {/* <button onClick={() => this.clickHandler()}>Click</button> */}
                <button onClick={this.clickHandler}>Click</button>
            </div>
        )
    }
}

export default EventBind
