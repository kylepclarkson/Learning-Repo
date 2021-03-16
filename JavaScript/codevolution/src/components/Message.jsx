import React, { Component } from 'react'
const buttonClicked = () => {
    this.state.message = 'Thank you for the sub!'
}
class Message extends Component {

    constructor() {
        super();
        this.state = {
            isSub: false,
            message: 'Please subscribe!',
        };
    }
    
    changeMessage() {
        if (!this.state.isSub){
            this.setState({
                isSub: true,
                message: 'Thank you for the sub!', 
            })
        } else {
            this.setState({
                isSub: false,
                message: 'Please sub!', 
            })
        }
    }

    render() {
        return (
            <div className="">
                <h1>
                    {this.state.message}
                </h1>
                <button onClick={() => this.changeMessage()}>{this.state.message}</button>
            </div>
        )
    }
}

export default Message;